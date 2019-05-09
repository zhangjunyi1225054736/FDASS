import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
from PIL import Image

from model.deeplab_multi_wn import DeeplabMulti
from model.discriminator_wn import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.SynthiaDataset import SynthiaDataSet
from dataset.cityscapes_dataset_pro import cityscapesDataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/data2/zhangjunyi/AdaptSegNet/data/RAND_CITYSCAPES'
DATA_LIST_PATH = 'dataset/SYNTHIA_list/SYNTHIA_imagelist_train.txt'
IGNORE_LABEL = 255
# INPUT_SIZE = '1280,720'
INPUT_SIZE = '1024,512'
DATA_DIRECTORY_TARGET = '/data2/zhangjunyi/AdaptSegNet/data/cityscapes'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train_18.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 120000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'model/DeepLab_resnet_pretrained_init.pth'
    # RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2500
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

TARGET = 'cityscapes'
SET = 'train'

cut_size = (1024, 564)

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--beta", type=float, default=0.95,
                        help="available options : DeepLab")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=1,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu, alpha=0.95):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    other_label = (label + 0).long()
    tem_pred = pred + 0
    if alpha != 0:
        tem = F.softmax(tem_pred, 1)
        value, posi = torch.max(F.softmax(tem_pred, 1), 1)
        soft_pred, _ = torch.max(F.softmax(tem_pred, 1), 1)
        other_label[soft_pred>alpha] = 255
    # print(torch.sum(soft_pred-255 == torch.zeros(soft_pred.size()).long()))
    criterion = CrossEntropy2d().cuda(gpu)
    # print("pred",pred.size())
    # print("soft_label",soft_label.size())
    # print(alpha)
    return criterion(pred, label), other_label


def set_cuda(model, model_D1, model_D2):
    model = model.cuda(1)
    model_D1 = model_D1.cuda(0)
    model_D2 = model_D2.cuda(0)
    return model, model_D1, model_D2

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def main():
    """Create the model and start the training."""

    gpu_id_2 = 1
    gpu_id_1 = 0
    
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    # gpu = args.gpu

    # Create network
    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            print("from url")
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            print("from restore")
            #saved_state_dict = torch.load(args.restore_from)
#             saved_state_dict = torch.load('/home/huangchangxin/AdaptMutilStage_gta5/resnet101-pretrained.pth')
            saved_state_dict = torch.load('/data2/zhangjunyi/snapshots/snapshots_syn/onlysyn/GTA5_80000.pth')
#             saved_state_dict = torch.load('/data2/zhangjunyi/snapshots/snapshots_syn2/syn_36/GTA5_7500.pth')
            # print('snapshots/GTA2Cityscapes_multi_wn/GTA5_10000.pth')
            # model.load_state_dict(saved_state_dict)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if (not i_parts[1] == 'layer5') and (not i_parts[0]=='fc'):
                new_params['.'.join(i_parts)] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)

    model.train()
    model.cuda(gpu_id_2)

    cudnn.benchmark = True

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes)
#     model_D2 = model_D1

    model_D1.train()
    model_D1.cuda(gpu_id_1)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        SynthiaDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)
    _, batch_last = trainloader_iter.__next__()

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     cut_size=cut_size,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)
    # print(args.num_steps * args.iter_size * args.batch_size, trainloader.__len__())

    targetloader_iter = enumerate(targetloader)
    _, batch_last_target = targetloader_iter.__next__()
    
    # for i in range(200):
    #     _, batch = targetloader_iter.__next__()
    # exit()

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    mse_loss = torch.nn.MSELoss()

    def upsample_(input_):
        return nn.functional.interpolate(input_, size=(input_size[1], input_size[0]), mode='bilinear', align_corners=False)

    def upsample_target(input_):
        return nn.functional.interpolate(input_, size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=False)

    interp = upsample_
    interp_target = upsample_target

    # labels for adversarial training
    source_label = 1
    target_label = -1
    mix_label = 0

    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0
        
        number1 = 0
        number2 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        
        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            def result_model(batch, interp_):
                images, labels, _, name = batch
                images = Variable(images).cuda(gpu_id_2)
                labels = Variable(labels.long()).cuda(gpu_id_1)
                pred1, pred2 = model(images)
                pred1 = interp_(pred1)
                pred2 = interp_(pred2)
                pred1_ = pred1.cuda(gpu_id_1)
                pred2_ = pred2.cuda(gpu_id_1)
                return pred1_, pred2_, labels

            beta = args.beta
            if i_iter == 0:
                print(beta)
            # train with source
            # _, batch = trainloader_iter.next()
            _, batch = trainloader_iter.__next__()
            _, batch_target = targetloader_iter.__next__()
            pred1, pred2, labels = result_model(batch, interp)
            loss_seg1, new_labels = loss_calc(pred1, labels, gpu_id_1, beta)
            labels = new_labels
            number1 = torch.sum(labels==255).item()
            loss_seg2, new_labels = loss_calc(pred2, labels, gpu_id_1, beta)
            loss = loss_seg2 + args.lambda_seg * loss_seg1
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_1 = loss_seg1.data.cpu().numpy() / args.iter_size
            loss_seg_2 = loss_seg2.data.cpu().numpy() / args.iter_size
            # print(loss_seg_1, loss_seg_2)
            
            pred1, pred2, labels = result_model(batch_target, interp_target)
            loss_seg1, new_labels = loss_calc(pred1, labels, gpu_id_1, beta)
            labels = new_labels
            number2 = torch.sum(labels==255).item()
            loss_seg2, new_lables = loss_calc(pred2, labels, gpu_id_1, beta)
            loss = loss_seg2 + args.lambda_seg * loss_seg1
            loss = loss / args.iter_size
            loss.backward()
#             break
            # print(number2)
            # exit()
            
            loss_seg_value1 += loss_seg1.data.cpu().numpy() / args.iter_size
            loss_seg_value2 += loss_seg2.data.cpu().numpy() / args.iter_size
            
            pred1_last_target, pred2_last_target, labels_last_target = result_model(batch_last_target, interp_target)
            pred1_target, pred2_target, labels_target = result_model(batch_target, interp_target)
            # exit()


            pred1_target_D = F.softmax((pred1_target), dim=1)
            pred2_target_D = F.softmax((pred2_target), dim=1)
            pred1_last_target_D = F.softmax((pred1_last_target), dim=1)
            pred2_last_target_D = F.softmax((pred2_last_target), dim=1)
            fake1_D = torch.cat((pred1_target_D, pred1_last_target_D), dim=1)
            fake2_D = torch.cat((pred2_target_D, pred2_last_target_D), dim=1)
            D_out_fake_1 = model_D1(fake1_D)
            D_out_fake_2 = model_D1(fake2_D)

            loss_adv_fake1 = mse_loss(D_out_fake_1,
                                       Variable(torch.FloatTensor(D_out_fake_1.data.size()).fill_(source_label)).cuda(
                                            gpu_id_1))

            loss_adv_fake2 = mse_loss(D_out_fake_2,
                                        Variable(torch.FloatTensor(D_out_fake_2.data.size()).fill_(source_label)).cuda(
                                            gpu_id_1))
                                            
            loss_adv_target1 = loss_adv_fake1
            loss_adv_target2 = loss_adv_fake2
            loss = args.lambda_adv_target1 * loss_adv_target1.cuda(gpu_id_1) + args.lambda_adv_target2 * loss_adv_target2.cuda(gpu_id_1)
            loss = loss / args.iter_size
            loss.backward()
            
            pred1, pred2, labels = result_model(batch, interp)
            pred1_target, pred2_target, labels_target = result_model(batch_target, interp_target)
            
            pred1_target_D = F.softmax((pred1_target), dim=1)
            pred2_target_D = F.softmax((pred2_target), dim=1)
            pred1_D = F.softmax((pred1), dim=1)
            pred2_D = F.softmax((pred2), dim=1)
            mix1_D = torch.cat((pred1_target_D, pred1_D), dim=1)
            mix2_D = torch.cat((pred2_target_D, pred2_D), dim=1)

            D_out_mix_1 = model_D1(mix1_D)
            D_out_mix_2 = model_D1(mix2_D)
            
            loss_adv_mix1 = mse_loss(D_out_mix_1,
                                       Variable(torch.FloatTensor(D_out_mix_1.data.size()).fill_(source_label)).cuda(
                                            gpu_id_1))

            loss_adv_mix2 = mse_loss(D_out_mix_2,
                                        Variable(torch.FloatTensor(D_out_mix_2.data.size()).fill_(source_label)).cuda(
                                            gpu_id_1))

            loss_adv_target1 = loss_adv_mix1*2
            loss_adv_target2 = loss_adv_mix2*2

            loss = args.lambda_adv_target1 * loss_adv_target1.cuda(gpu_id_1) + args.lambda_adv_target2 * loss_adv_target2.cuda(gpu_id_1)
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy() / args.iter_size
            loss_adv_target_value1 += loss_adv_target2.data.cpu().numpy() / args.iter_size
            
            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            # train with source            
            pred1_last, pred2_last, labels_last = result_model(batch_last, interp)

            
            pred1 = pred1.detach().cuda(gpu_id_1)
            pred2 = pred2.detach().cuda(gpu_id_1)
            pred1_target = pred1_target.detach().cuda(gpu_id_1)
            pred2_target = pred2_target.detach().cuda(gpu_id_1)
            pred1_last = pred1_last.detach().cuda(gpu_id_1)
            pred2_last = pred2_last.detach().cuda(gpu_id_1)
            
            pred1_D = F.softmax((pred1), dim=1)
            pred2_D = F.softmax((pred2), dim=1)
            pred1_last_D = F.softmax((pred1_last), dim=1)
            pred2_last_D = F.softmax((pred2_last), dim=1)
            pred1_target_D = F.softmax((pred1_target), dim=1)
            pred2_target_D = F.softmax((pred2_target), dim=1)

            real1_D = torch.cat((pred1_D, pred1_last_D), dim=1)
            real2_D = torch.cat((pred2_D, pred2_last_D), dim=1)
            mix1_D_ = torch.cat((pred1_last_D, pred1_target_D), dim=1)
            mix2_D_ = torch.cat((pred2_last_D, pred2_target_D), dim=1)

            D_out1_real = model_D1(real1_D)
            D_out2_real = model_D1(real2_D)
            D_out1_mix = model_D1(mix1_D_)
            D_out2_mix = model_D1(mix2_D_)

            loss_D1 = mse_loss(D_out1_real,
                              Variable(torch.FloatTensor(D_out1_real.data.size()).fill_(source_label)).cuda(gpu_id_1))

            loss_D2 = mse_loss(D_out2_real,
                               Variable(torch.FloatTensor(D_out2_real.data.size()).fill_(source_label)).cuda(gpu_id_1))

            loss_D3 = mse_loss(D_out1_mix,
                              Variable(torch.FloatTensor(D_out1_mix.data.size()).fill_(mix_label)).cuda(gpu_id_1))

            loss_D4 = mse_loss(D_out2_mix,
                               Variable(torch.FloatTensor(D_out2_mix.data.size()).fill_(mix_label)).cuda(gpu_id_1))

            loss_D1 = (loss_D1 + loss_D3) / args.iter_size / 2
            loss_D2 = (loss_D2 + loss_D4) / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value1 += loss_D2.data.cpu().numpy()

            # train with target

            pred1 = pred1.detach().cuda(gpu_id_1)
            pred2 = pred2.detach().cuda(gpu_id_1)
            pred1_target = pred1_target.detach().cuda(gpu_id_1)
            pred2_target = pred2_target.detach().cuda(gpu_id_1)
            pred1_last_target = pred1_last_target.detach().cuda(gpu_id_1)
            pred2_last_target = pred2_last_target.detach().cuda(gpu_id_1)

            pred1_D = F.softmax((pred1), dim=1)
            pred2_D = F.softmax((pred2), dim=1)
            pred1_last_target_D = F.softmax((pred1_last_target), dim=1)
            pred2_last_target_D = F.softmax((pred2_last_target), dim=1)
            pred1_target_D = F.softmax((pred1_target), dim=1)
            pred2_target_D = F.softmax((pred2_target), dim=1)

            fake1_D_ = torch.cat((pred1_last_target_D, pred1_target_D), dim=1)
            fake2_D_ = torch.cat((pred2_last_target_D, pred2_target_D), dim=1)
            mix1_D__ = torch.cat((pred1_D, pred1_last_target_D), dim=1)
            mix2_D__ = torch.cat((pred2_D, pred2_last_target_D), dim=1)

            D_out1 = model_D1(fake1_D_)
            D_out2 = model_D1(fake2_D_)
            D_out3 = model_D1(mix1_D__)
            D_out4 = model_D1(mix2_D__)

            loss_D1 = mse_loss(D_out1,
                              Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda(gpu_id_1))

            loss_D2 = mse_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda(gpu_id_1))
            
            loss_D3 = mse_loss(D_out3,
                              Variable(torch.FloatTensor(D_out3.data.size()).fill_(mix_label)).cuda(gpu_id_1))

            loss_D4 = mse_loss(D_out4,
                               Variable(torch.FloatTensor(D_out4.data.size()).fill_(mix_label)).cuda(gpu_id_1))

            loss_D1 = (loss_D1+loss_D3) / args.iter_size / 2
            loss_D2 = (loss_D2+loss_D4) / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            batch_last, batch_last_target = batch, batch_target
            loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value1 += loss_D2.data.cpu().numpy()
            
            
            

        optimizer.step()
        optimizer_D1.step()

        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f}, number1 = {8}, number2 = {9}'.format(
            i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2, number1, number2))

        if i_iter >= args.num_steps_stop - 1:
            print ('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1.pth'))
            break

        if i_iter % args.save_pred_every == 0:
            print ('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))


if __name__ == '__main__':
    main()
