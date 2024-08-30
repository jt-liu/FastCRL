import argparse
import gc
import os
import sys
import time
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
sys.path.append("..")
from Keypoint_CRL.models.seg_unet import SegUNet
from Keypoint_CRL.utils import *
from Keypoint_CRL.loss import JointsSILoss
from Keypoint_CRL.datasets.dataloader_keypoint import KeypointDataset

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='UltraSonicSegment_CRL')
parser.add_argument('--lr', type=float, default=0.000375, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--val_batch_size', type=int, default=16, help='testing batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--resume', default='', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=3407, metavar='S', help='random seed (default: 1)')
parser.add_argument('--num_class', type=int, default=3, help='the number of segment class')
parser.add_argument('--offset_radius', type=int, default=4, help='the radius of offset maps (origin resolution)')

parser.add_argument('--datapath', default='../Keypoint_CRL/Data_3Hospital/', help='data path')
parser.add_argument('--trainlist', default='../Keypoint_CRL/filenames/trainlist.txt', help='training list')
parser.add_argument('--vallist', default='../Keypoint_CRL/filenames/vallist.txt', help='validate list')
parser.add_argument('--testlist', default='../Keypoint_CRL/filenames/testlist.txt', help='testing list')
parser.add_argument('--ckptdir', default='../Keypoint_CRL/ckpts/', help='the directory to save checkpoints')
parser.add_argument('--loadckpt', default='', help='load the weights from a specific checkpoint')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--summary_freq', type=int, default=100, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=50, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
train_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
args.ckptdir = os.path.join(args.ckptdir, str(train_time))
os.makedirs(args.ckptdir, exist_ok=True)

# dataset, dataloader
train_dataset = KeypointDataset(args.datapath, args.trainlist, training=True, imgz=(480, 640),
                                radius=args.offset_radius)
val_dataset = KeypointDataset(args.datapath, args.testlist, training=False, imgz=(480, 640),
                              radius=args.offset_radius)
test_dataset = KeypointDataset(args.datapath, args.testlist, training=False, imgz=(480, 640),
                               radius=args.offset_radius)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                            drop_last=True)
ValImgLoader = DataLoader(val_dataset, args.val_batch_size, shuffle=False, num_workers=8, pin_memory=True,
                          drop_last=False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True,
                           drop_last=False)

torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SegUNet(num_class=args.num_class)
from thop import profile

input1 = torch.randn(1, 3, 480, 640)
flops, params = profile(model, inputs=(input1,))
print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
print('Params = ' + str(params / 1000 ** 2) + 'M')
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

# model = torch.nn.DataParallel(model)
model = model.to(device)

optimizer = optim.AdamW(params=model.parameters(), weight_decay=0.01, lr=args.lr)
# optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(TrainImgLoader), cycle_momentum=True,
                                                base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
                                                div_factor=37.5, final_div_factor=10)

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.ckptdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.ckptdir, all_saved_ckpts[-1])
    print("Loading the latest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    scheduler.load_state_dict(state_dict['scheduler'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("Loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'], strict=True)

print("Start at epoch {}".format(start_epoch))


def train():
    prefix = 'KeyPointCRL_'
    logger = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.batch_size),
                           flush_secs=60)
    best_checkpoint_acc = 999
    for epoch_idx in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()
        train_loop = tqdm(enumerate(TrainImgLoader), total=len(TrainImgLoader),
                          desc=f'train epoch[{epoch_idx}/{args.epochs}]')
        time.sleep(1)
        losses = AverageMeter()
        # training
        for batch_idx, sample in train_loop:
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss = train_sample(sample, do_summary, global_step)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            loss = tensor2float(loss)
            losses.update(loss)
            train_loop.set_postfix({'loss': round(loss, 2)})
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                               'scheduler': scheduler.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.ckptdir, epoch_idx))
        # scheduler.step()
        gc.collect()

        # validating
        val_loop = tqdm(enumerate(ValImgLoader), total=len(ValImgLoader),
                        desc=f'val epoch[{epoch_idx}/{args.epochs}]')
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in val_loop:
            global_step = len(ValImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss = test_sample(sample, do_summary, global_step, avg_test_scalars)
            val_loop.set_postfix({'loss': round(loss, 2)})
        avg_test_scalars = avg_test_scalars.mean()

        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)

        # saving new best checkpoint
        if avg_test_scalars['acc'] < best_checkpoint_acc:
            best_checkpoint_acc = avg_test_scalars['acc']
            print("Overwriting best checkpoint")
            torch.save(model.state_dict(), "{}/best.ckpt".format(args.ckptdir))

        gc.collect()


# train one sample
def train_sample(sample, do_summary, global_step):
    model.train()

    images, labels = sample['image'], sample['label']
    images = images.to(device)
    labels = labels.to(device)

    offset_maps_x, offset_maps_y = sample["offset_x"], sample["offset_y"]
    offset_maps_x = offset_maps_x.to(device)
    offset_maps_y = offset_maps_y.to(device)

    pred, offset_x, offset_y = model(images)
    # pred = model(images)
    # JointsSILoss = nn.MSELoss()
    loss1 = JointsSILoss(pred, labels)
    loss2 = JointsSILoss(offset_x, offset_maps_x)
    loss3 = JointsSILoss(offset_y, offset_maps_y)
    loss = torch.sqrt(loss1 + loss2 * 0.1 + loss3 * 0.1)
    # loss = loss1
    # if do_summary:
    #     images = F.interpolate(images, offset_maps_x.shape[2:])
    #     mask_ests = torch.sum(pred, dim=1).unsqueeze(1).type(torch.float)
    #     mask_gt = torch.sum(labels, dim=1).unsqueeze(1).type(torch.float)
    #     acc = accuracy(np.array(pred.detach().cpu()), np.array(labels.detach().cpu()))[1]
    #     scalar_outputs = {"loss": loss, "LR": optimizer.param_groups[0]['lr'], "acc": acc}
    #     image_outputs = {
    #         "mask_est": mask_ests * 255 * 0.7 + images * 255 * 0.3,
    #         "mask_gt": mask_gt * 255 * 0.7 + images * 255 * 0.3,
    #         'head_point': labels[:, 0, :, :].unsqueeze(1) * 255 * 0.7 + images * 255 * 0.3,
    #         'ass_point': labels[:, 1, :, :].unsqueeze(1) * 255 * 0.7 + images * 255 * 0.3,
    #         'rotate_point': labels[:, 2, :, :].unsqueeze(1) * 255 * 0.7 + images * 255 * 0.3,
    #         'head_offset': (offset_x[:, 0, :, :].unsqueeze(1) + 1) / 2 * 255 * 0.7 + images * 255 * 0.3,
    #         'ass_offset': (offset_x[:, 1, :, :].unsqueeze(1) + 1) / 2 * 255 * 0.7 + images * 255 * 0.3,
    #         'rotate_offset': (offset_x[:, 2, :, :].unsqueeze(1) + 1) / 2 * 255 * 0.7 + images * 255 * 0.3,
    #         'offset_x': (torch.sum(offset_maps_x, dim=1).unsqueeze(1) + 1) / 2 * 255 * 0.7 + images * 255 * 0.3,
    #         'offset_y': (torch.sum(offset_maps_y, dim=1).unsqueeze(1) + 1) / 2 * 255 * 0.7 + images * 255 * 0.3,
    #         "error": torch.abs(mask_ests - mask_gt) * 255, "image": images
    #     }
    #     save_scalars(logger, 'train', scalar_outputs, global_step)
    #     save_images(logger, 'train', image_outputs, global_step)
    #     del scalar_outputs, image_outputs

    return loss


# test one sample
@make_nograd_func
def test_sample(sample, do_summary, global_step, avg_test_scalars):
    model.eval()

    images, labels = sample['image'], sample['label']
    images = images.to(device)
    labels = labels.to(device)

    offset_maps_x, offset_maps_y = sample["offset_x"], sample["offset_y"]
    offset_maps_x = offset_maps_x.to(device)
    offset_maps_y = offset_maps_y.to(device)

    pred, offset_x, offset_y = model(images)
    # pred = model(images)
    # JointsSILoss = nn.MSELoss()
    loss1 = JointsSILoss(pred, labels)
    loss2 = JointsSILoss(offset_x, offset_maps_x)
    loss3 = JointsSILoss(offset_y, offset_maps_y)
    loss = torch.sqrt(loss1 + loss2 * 0.1 + loss3 * 0.1)
    # loss = loss1
    loss = tensor2float(loss)

    # err = distance_points(pred, sample['points'])
    err = offset_distance(pred, offset_x, offset_y, sample['points'], radius=args.offset_radius, scale_factor=4)
    # a1, a2, a3, angle = crl_heatmap_acc(pred, sample['points'])
    a1, a2, a3, angle = crl_mask_acc(pred, offset_x, offset_y, sample['points'], radius=args.offset_radius, scale_factor=4)
    scalar_outputs = {"loss": loss, "acc": err, "a1": a1, "a2": a2, "a3": a3, "angle": angle}
    avg_test_scalars.update(scalar_outputs)
    # if do_summary:
    #     images = F.interpolate(images, labels.shape[2:])
    #     mask_ests = torch.sum(pred, dim=1).unsqueeze(1).type(torch.float)
    #     mask_gt = torch.sum(labels, dim=1).unsqueeze(1).type(torch.float)
    #     image_outputs = {"mask_est": mask_ests * 255 * 0.7 + images * 255 * 0.3,
    #                      "mask_gt": mask_gt * 255 * 0.7 + images * 255 * 0.3,
    #                      'head_point': labels[:, 0, :, :].unsqueeze(1) * 255 * 0.7 + images * 255 * 0.3,
    #                      'ass_point': labels[:, 1, :, :].unsqueeze(1) * 255 * 0.7 + images * 255 * 0.3,
    #                      'rotate_point': labels[:, 2, :, :].unsqueeze(1) * 255 * 0.7 + images * 255 * 0.3,
    #                      "error": torch.abs(mask_ests - mask_gt) * 255, "image": images}
    #     save_images(logger, 'test', image_outputs, global_step)
    #     save_scalars(logger, 'test', scalar_outputs, global_step)
    #     del image_outputs, scalar_outputs

    return loss


@make_nograd_func
def testing():
    import matplotlib.pyplot as plt

    torch.cuda.empty_cache()
    time.sleep(1)
    load_val_ckpt = os.path.join(args.ckptdir, 'best.ckpt')
    val_state_dict = torch.load(load_val_ckpt)
    model.load_state_dict(val_state_dict, strict=False)
    model.eval()
    times = []
    err_list = []
    a1_list = []
    a2_list = []
    a3_list = []
    angle_list = []
    count_err = 0
    for batch_idx, sample in enumerate(TestImgLoader):
        images, labels = sample['image'], sample['label']
        images = images.to(device)
        labels = labels.to(device)

        start = time.time()
        mask_pred, offset_x, offset_y = model(images)
        # mask_pred = model(images)
        end = time.time()
        infer_time = end - start
        times.append(infer_time)
        # err = distance_points(mask_pred, sample['points'])
        err = offset_distance(mask_pred, offset_x, offset_y, sample['points'], radius=args.offset_radius)
        err_list.append(err)
        # a1, a2, a3, angle = crl_heatmap_acc(mask_pred, sample['points'])
        a1, a2, a3, angle = crl_mask_acc(mask_pred, offset_x, offset_y, sample['points'], radius=args.offset_radius)
        a1_list.append(a1)
        a2_list.append(a2)
        a3_list.append(a3)
        angle_list.append(angle)
        if a1 == 0.0:
            print(err)
            print(a1, a2, a3)
            # count_err += 1
            # mask_ests = mask_pred[:, 0, :, :] + mask_pred[:, 1, :, :] + mask_pred[:, 2, :, :]
            # mask_ests = mask_ests.unsqueeze(1).type(torch.float)
            # mask_gt = labels[:, 0, :, :] + labels[:, 1, :, :] + labels[:, 2, :, :]
            # mask_gt = mask_gt.unsqueeze(1).type(torch.float)
            # images = F.interpolate(images, labels.shape[2:])
            # image_outputs = {"mask_est": mask_ests * 255 * 0.7 + images * 255 * 0.3,
            #                  "mask_gt": mask_gt * 255 * 0.7 + images * 255 * 0.3,
            #                  'head_point': labels[:, 0, :, :].unsqueeze(1) * 255 * 0.7 + images * 255 * 0.3,
            #                  'ass_point': labels[:, 1, :, :].unsqueeze(1) * 255 * 0.7 + images * 255 * 0.3,
            #                  'rotate_point': labels[:, 2, :, :].unsqueeze(1) * 255 * 0.7 + images * 255 * 0.3,
            #                  'head_pred': mask_pred[:, 0, :, :].unsqueeze(1) * 255 * 0.7 + images * 255 * 0.3,
            #                  'ass_pred': mask_pred[:, 1, :, :].unsqueeze(1) * 255 * 0.7 + images * 255 * 0.3,
            #                  'rotate_pred': mask_pred[:, 2, :, :].unsqueeze(1) * 255 * 0.7 + images * 255 * 0.3,
            #                  "error": torch.abs(mask_ests - mask_gt) * 255, "image": images}
            # save_images(logger, 'val_err', image_outputs, count_err)
            # del image_outputs
    test_set_length = len(err_list)
    print('test set length', test_set_length)
    print('max(err_list)', max(err_list))
    print('average err', sum(err_list) / test_set_length)
    print('a1: {}, a2: {}, a3: {}, angle: {}'.format(sum(a1_list) / test_set_length,
                                                     sum(a2_list) / test_set_length,
                                                     sum(a3_list) / test_set_length,
                                                     sum(angle_list) / test_set_length))
    print('time: ', sum(times) / test_set_length)
    plt.figure(figsize=(23, 10))
    nums, bins, patches = plt.hist(err_list, bins=15, edgecolor='k', density=False)
    plt.xticks(bins, bins)
    xticks = plt.gca().get_xticks()
    plt.gca().set_xticklabels(['{}'.format(round(x, 2)) for x in xticks])
    plt.savefig('key points accuracy')
    pre_dict = {}
    val_model = SegUNet(num_class=args.num_class)
    model_dict = val_model.state_dict()
    for k, v in val_state_dict.items():
        k = k.replace('module.', '')
        pre_dict[k] = v
    model_dict.update(pre_dict)
    torch.save(model_dict, os.path.join(args.ckptdir, 'best_keypoint_CRL.ckpt'))


if __name__ == '__main__':
    train()
    testing()
