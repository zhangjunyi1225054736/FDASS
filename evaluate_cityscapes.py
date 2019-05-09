import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
from packaging import version

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = '/data2/zhangjunyi/AdaptSegNet/data/cityscapes'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
# DATA_LIST_PATH = './dataset/cityscapes_list/train_1.txt'
SAVE_PATH = '/data2/zhangjunyi/result/syn18_nostage_30'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
SET = 'val'
# SET = 'train'

MODEL = 'DeeplabMulti'

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
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=3,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()

def Upsample_function(input):
    return nn.functional.interpolate(input, size=(1024, 2048), mode='bilinear', align_corners=True)

def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    gpu0 = 2
    torch.cuda.manual_seed(1337)
    torch.cuda.set_device(2)

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    for i in range(5):
        if not os.path.exists(args.save + '/' + str(i)):
            os.makedirs(args.save + '/' + str(i))

    if args.model == 'DeeplabMulti':
        model = DeeplabMulti(num_classes=args.num_classes)
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG
    print("begin")

    if args.restore_from[:4] == 'http' :
        print("1112222")
        #saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        print("2222222", gpu0)
        # saved_state_dict = torch.load(args.restore_from)
        print(args.restore_from)
        model.load_state_dict(torch.load(args.restore_from))
    model.cuda(gpu0)
    # print(sys.getsizeof(model))
    # model.eval()
    # exit()

    testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024,512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=1, shuffle=False, pin_memory=True)


    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        # interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
        interp = Upsample_function

    else:
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear')
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            if index % 100 == 0:
                print ('%d processd' % index)
            image, labels, _, name = batch
            image = Variable(image).cuda(gpu0)
            final = []
            if args.model == 'DeeplabMulti':
                output1, output2 = model(image)
                output1 = F.softmax(output1, 1)
                output2 = F.softmax(output2, 1)
                for i in [0, 3, 7, 10]:
                    final_output = i/10.0*output1 + (10.0-i)/10.0*output2
                    output = interp(final_output).cpu().data[0].numpy()
                    final.append(output)
                    break
                labels = labels.cpu().data[0].numpy()
            elif args.model == 'DeeplabVGG':
                output = model(Variable(image, volatile=True).cuda(gpu0))
                output = interp(output).cpu().data[0].numpy()

            name = name[0].split('/')[-1]
            # labels_col = colorize_mask(labels)
            # labels_col.save('%s/%s_real.png' % (args.save, name.split('.')[0]))
            for i in range(4):
                output = final[i]
                output = output.transpose(1,2,0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                output_col = colorize_mask(output)
                output = Image.fromarray(output)


                output.save('%s/%s/%s' % (args.save, str(i), name))
                output_col.save('%s/%s/%s_color.png' % (args.save, str(i), name.split('.')[0]))
                break

            # i = 4
            # output1 = interp(output1).cpu().data[0].numpy()
            # output2 = interp(output2).cpu().data[0].numpy()
            # output1 = output1.transpose(1,2,0)
            # output2 = output2.transpose(1,2,0)
            # output_max = np.max(output1, axis=2)
            # output = np.asarray(np.argmax(output2, axis=2), dtype=np.uint8)
            # output1 = np.asarray(np.argmax(output1, axis=2), dtype=np.uint8)
            # output[output_max>0.95] = output1[output_max>0.95]
            # output_col = colorize_mask(output)
            # output = Image.fromarray(output)

            # output.save('%s/%s/%s' % (args.save, str(i), name))
            # output_col.save('%s/%s/%s_color.png' % (args.save, str(i), name.split('.')[0]))

if __name__ == '__main__':
    main()
