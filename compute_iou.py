# import numpy as np
# import argparse
# import json
# from PIL import Image
# from os.path import join


# def fast_hist(a, b, n):
#     k = (a >= 0) & (a < n)
#     # print(np.sum(a==b))
#     # for i in range(19):
#     #     print(np.sum((a==i) & (b==i)), np.sum((a!=i) & (b==i)), np.sum((a==i) & (b!=i)), np.sum((a!=i) & (b!=i)))
#     # exit()
#     return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


# def per_class_iu(hist):
#     return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


# def label_mapping(input, mapping):
#     output = np.copy(input)
#     for ind in range(len(mapping)):
#         output[input == mapping[ind][0]] = mapping[ind][1]
#     return np.array(output, dtype=np.int64)


# def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):
#     """
#     Compute IoU given the predicted colorized images and 
#     """
#     with open(join(devkit_dir, 'info.json'), 'r') as fp:
#       info = json.load(fp)
#     num_classes = np.int(info['classes'])
#     print('Num classes', num_classes)
#     name_classes = np.array(info['label'], dtype=np.str)
#     mapping = np.array(info['label2train'], dtype=np.int)
#     hist = np.zeros((num_classes, num_classes))

#     # image_path_list = join(devkit_dir, 'val.txt')
#     # label_path_list = join(devkit_dir, 'label.txt')
#     # gt_imgs = open(label_path_list, 'r').read().splitlines()
#     # gt_imgs = [join(gt_dir, x) for x in gt_imgs]
#     # pred_imgs = open(image_path_list, 'r').read().splitlines()
#     # pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

#     image_path_list = join(devkit_dir, 'train_1.txt')
#     label_path_list = join(devkit_dir, 'label.txt')
#     pred_imgs = open(image_path_list, 'r').read().splitlines()
#     gt_imgs = [x.split('_')[0] + "_" + x.split('_')[1] + "_" + x.split('_')[2] + "_gtFine_color.png" for x in pred_imgs]
#     pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]
#     # gt_imgs = open(label_path_list, 'r').read().splitlines()
#     gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    
#     for ind in range(len(gt_imgs)):
#         length = len(pred_imgs[ind])
#         pred_imgs[ind] = pred_imgs[ind][:length-16] + "_gtFine_color.png"
#         pred = np.array(Image.open(pred_imgs[ind]))
#         label = np.array(Image.open(gt_imgs[ind]))
#         label = label_mapping(label, mapping)
#         if len(label.flatten()) != len(pred.flatten()):
#             print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
#             continue
#         hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
#         if ind > 0 and ind % 10 == 0:
#             print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    
#     mIoUs = per_class_iu(hist)
#     for ind_class in range(num_classes):
#         print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
#     print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
#     return mIoUs

#     # for ind in range(len(gt_imgs)):
#     #     length = len(pred_imgs[ind])
#     #     pred_imgs[ind] = pred_imgs[ind][:length-16] + "_gtFine_color.png"
#     #     pred_path = pred_imgs[ind][:length-16] + "_gtFine_color_color.png"
#     #     real_path = pred_imgs[ind][:length-16] + "_gtFine_color_real.png"
        
#     #     pred = np.array(Image.open(pred_path))
#     #     label = np.array(Image.open(real_path))
#     #     # print(pred.shape, label.shape)
#     #     # print(np.max(pred), np.sum(pred==label))
#     #     # print(num_classes)
#     #     # print(mapping)
#     #     # exit()
#     #     # mapping = [[i, i] for i in range(19)]
#     #     # label = label_mapping(label, mapping)
#     #     if len(label.flatten()) != len(pred.flatten()):
#     #         print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
#     #         continue
#     #     hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
#     #     if ind > 0 and ind % 10 == 0:
#     #         print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    
#     # mIoUs = per_class_iu(hist)
#     # for ind_class in range(num_classes):
#     #     print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
#     # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
#     # return mIoUs


# def main(args):
#    compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('gt_dir', type=str, help='directory which stores CityScapes val gt images')
#     parser.add_argument('pred_dir', type=str, help='directory which stores CityScapes val pred images')
#     parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')
#     args = parser.parse_args()
#     main(args)


import numpy as np
import argparse
import json
from PIL import Image
from os.path import join


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val.txt')
    label_path_list = join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]
    print(pred_dir)
    # gt_imgs = open(label_path_list, 'r').read().splitlines()
    # gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    # pred_imgs = open(image_path_list, 'r').read().splitlines()
    # pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]
    # mapping_data = {0:7, 1:11, 2:2, 3:21, 4:8, 5:17, 6:20, 7:24, 8:23, 9:26, 10:19, 11:5, 12:33, 13:25, 14:13, 15:12, 16:20, 17:5, 18:9}
    for ind in range(len(gt_imgs)):
        length = len(pred_imgs[ind])
        pred_imgs[ind] = pred_imgs[ind][:length-16] + "_gtFine_labelIds_color.png"
        # label_path = pred_imgs[ind][:length-16] + "_gtFine_color_real.png"
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        # label2 = np.array(Image.open(label_path))
        # for x in mapping_data:
        #     pred[pred==x] = mapping_data[x]
        # print(pred.shape)
        # exit()
        # a, b = label.shape
        # for x in range()
        # abc = {}
        # for i in range(420):
        #     for j in range(840):
        #         ii = i*1024//420
        #         jj = j*2048//840
        #         if label2[i][j] in abc:
        #             if label[ii][jj] in abc[label2[i][j]]:
        #                 abc[label2[i][j]][label[ii][jj]] += 1
        #             else:
        #                 abc[label2[i][j]][label[ii][jj]] = 1
        #         else:
        #             abc[label2[i][j]] = {label[ii][jj]:1}
        # print(abc)
        # print(np.max(label), np.min(label))
        # print(np.max(label2), np.min(label2))
        # print(label.shape, label2.shape)
        # print(np.sum(label == label2))
        # exit()
        # if np.max(label) == 18:
        #     exit()
        # else:
        #     continue
        label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU 19 class: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    print('-------------------')
    total_iou = 0
    for ind_class in range(num_classes):
        #if ind_class !=3 and ind_class !=4 and ind_class !=5 and ind_class !=9 and ind_class !=14 and ind_class !=16:
          if ind_class not in (3,4,5,9,14,16):
            print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            total_iou = total_iou + round(mIoUs[ind_class] * 100,2)
    #print(total_iou)
    print('===> mIoU 13 class:', round(total_iou / 13,2))
    return mIoUs


def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_dir', type=str, help='directory which stores CityScapes val gt images')
    parser.add_argument('pred_dir', type=str, help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')
    args = parser.parse_args()
    main(args)


