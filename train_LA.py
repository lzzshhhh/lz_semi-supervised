import argparse
import logging
import os
import random
import shutil
import sys
import time
import torch.nn.functional as F
import numpy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss, MSELoss, L1Loss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

# from networks.discriminator import FC3DDiscriminator
import test_util
from dataloaders.la_heart import LAHeart, RandomCrop, ToTensor, TwoStreamBatchSampler, SingleStreamBatchSampler
from networks.VNet_Son1 import VNet
# from networks.vnet_sdf_MC_Srecon import VNet
from utils import ramps, losses, metrics
from utils.util import compute_sdf

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/Son1', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=8000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=2,help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float, default=1e-4,
                    help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='random seed')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--consistency_weight', type=float, default=0.1,
                    help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--beta', type=float,
                    default=0.3,help='balance factor to control regional and sdm loss')
# parser.add_argument('--gamma', type=float, default=0.2,help='balance factor to control supervised and consistency loss')


# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="kl", help='consistency_type')
parser.add_argument('--with_cons', type=str,
                    default="without_cons", help='with or without consistency')
parser.add_argument('--consistency', type=float,
                    default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
args = parser.parse_args()


train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + \
                "_{}labels_beta_{}/".format(
                    args.labelnum, args.beta)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.cuda.empty_cache()  # 清理未使用的显存

batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
T = 0.1
patch_size = (160, 128, 80)


# (112, 112, 80) (160, 128, 80)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__','/data']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes - 1, normalization='batchnorm', has_dropout=True,
                   has_skipconnect=False)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    model = create_model()

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',  # train/val split
                       transform=transforms.Compose([
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    labelnum = args.labelnum  # default 16
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, 80))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    # batch_sampler = SingleStreamBatchSampler(labeled_idxs, batch_size)


    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.99), amsgrad=False)
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()
    l1_loss = L1Loss()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path + '/log_{}_iretations'.format(args.max_iterations))
    logging.info("{} itertations per epoch".format(len(trainloader)))

    best_dice = 0
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=50)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):

            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            #outputs_seg, outputs_tanh, outputs_Sre = model(volume_batch)
            out_seg, out_sdm1, out_seg1, out_seg2, out_sdm2 = model(volume_batch)
            out_seg_soft = torch.sigmoid(out_seg)
            # print(out_seg1.shape)

            # calculate the loss
            with torch.no_grad():
                gt_sdm = compute_sdf(label_batch[:].cpu().numpy(), out_seg[:labeled_bs, 0, ...].shape)
                # gt_sdm1 = compute_sdf(label_batch[:].cpu().numpy(), out_seg1[:labeled_bs, 0, ...].shape)
                # gt_sdm2 = compute_sdf(label_batch[:].cpu().numpy(), out_seg2[:labeled_bs, 0, ...].shape)
                # gt_Tsdm = -gt_sdm + numpy.sign(gt_sdm)
                gt_sdm = torch.from_numpy(gt_sdm).float().cuda()
            # gt_sdm1 = torch.from_numpy(gt_sdm).float().cuda()
            # gt_sdm2 = torch.from_numpy(gt_sdm).float().cuda()

            loss_sdf = (mse_loss(out_sdm1[:labeled_bs, 0, ...], gt_sdm)+mse_loss(out_sdm2[:labeled_bs, 0, ...], gt_sdm)) #带标签数据GT转换成sdf与预测的损失
            loss_seg = ce_loss(out_seg[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())#分割预测交叉熵损失
            loss_seg_dice = losses.dice_loss(out_seg_soft[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)#带标签数据分割预测与GT的dice损失

            # gt_softmask = torch.sigmoid(-150 * gt_sdm)
                # dis_to_mask = torch.sigmoid(-150 * gt_sdm)
            out_sdm1_to_mask = torch.sigmoid(-150 * out_sdm1)
            out_sdm2_to_mask = torch.sigmoid(-150 * out_sdm2)

            # print("out_seg1.shape:", out_seg1.shape)  # [N, C, D, H, W]
            # print("label_batch.shape:", label_batch.shape)
            # print("label_batch min/max:", label_batch.min().item(), label_batch.max().item())
            # print("label_batch dtype:", label_batch.dtype)
            # loss_seg1=F.binary_cross_entropy(out_seg1[:labeled_bs], label_batch[:labeled_bs])
            out_seg1_soft = F.softmax(out_seg1, dim=1)
            # print("out_seg1_soft shape:", out_seg1_soft.shape)
            # print("out_seg1 shape:", out_seg1.shape)
            out_seg1_loss_seg_dice = losses.dice_loss(out_seg1_soft[:labeled_bs, 0, :, :, :],(out_seg1[:labeled_bs] == 1).float())

            # loss_seg2=F.binary_cross_entropy(out_seg2[:labeled_bs], label_batch[:labeled_bs])
            out_seg2_soft = F.softmax(out_seg2, dim=1)
            out_seg2_loss_seg_dice = losses.dice_loss(out_seg2_soft[:labeled_bs, 0, :, :, :],(out_seg2[:labeled_bs] == 1))

            if out_seg1_loss_seg_dice < out_seg2_loss_seg_dice:
                Good_student = 0
            else:
                Good_student = 1

            # out_seg1_soft2 = F.softmax(out_seg1, dim=1)
            # out_seg2_soft2 = F.softmax(out_seg2, dim=1)

            out_seg1_clone = out_seg1_soft[labeled_bs:, :, :, :, :].clone().detach()
            out_seg2_clone = out_seg2_soft[labeled_bs:, :, :, :, :].clone().detach()
            out_seg1_clone1 = torch.pow(out_seg1_clone, 1 / T)
            out_seg2_clone1 = torch.pow(out_seg2_clone, 1 / T)
            out_seg1_clone2 = torch.sum(out_seg1_clone1, dim=1, keepdim=True)
            out_seg2_clone2= torch.sum(out_seg2_clone1, dim=1, keepdim=True)
            out_seg1_PLable = torch.div(out_seg1_clone1, out_seg1_clone2)
            out_seg2_PLable = torch.div(out_seg2_clone1, out_seg2_clone2)

            if Good_student == 0:
                Plabel = out_seg1_PLable
                sdm_to_mask = out_sdm1_to_mask
                out_sdm=out_sdm1

            if Good_student == 1:
                Plabel = out_seg2_PLable
                sdm_to_mask = out_sdm2_to_mask
                out_sdm=out_sdm2
            consistency_loss = torch.mean((sdm_to_mask - out_seg_soft) ** 2)
            supervised_loss = loss_seg_dice + args.beta * loss_sdf
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dc = metrics.dice(torch.argmax(out_seg[:labeled_bs], dim=1), label_batch[:labeled_bs])

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_hausdorff', loss_sdf, iter_num)
            writer.add_scalar('loss/consistency_weight',consistency_weight, iter_num)
            writer.add_scalar('loss/consistency_loss',consistency_loss, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_consis: %f, loss_sdf: %f, loss_Seg: %f, loss_dice: %f, consis_weight: %f' %
                (iter_num, loss.item(), consistency_loss.item(), loss_sdf.item(),
                 loss_seg.item(), loss_seg_dice.item(), consistency_weight))

            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = out_seg_soft[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = sdm_to_mask[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/sdm2Mask', grid_image, iter_num)

                image = out_sdm[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/sdmMap', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

                image = gt_sdm[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_sdm2Map',
                                 grid_image, iter_num)

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num >= 800 / (args.labeled_bs/2)  and iter_num % (200 / (args.labeled_bs / 2)) == 0:
                model.eval()
                dice_sample = test_util.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                     stride_xy=18, stride_z=4, data_path=args.root_path, nms=1)

                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, 'maxiretation_{}_iter_{}_dice_{}.pth'.format(max_iterations,iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, 'maxiretation_{}_best_model.pth'.format(max_iterations))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
