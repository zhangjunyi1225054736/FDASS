import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import json

class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        print(1, len(self.img_ids))
        #exit()
        if not max_iters==None:
           self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        with open('dataset/cityscapes_list/info.json', 'r') as fp:
            self.info = json.load(fp)
        # print(self.info)
        # exit()

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            if name == "":
                continue
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            tmp = name.split('_')
            name = tmp[0]+'_'+tmp[1]+'_'+tmp[2]+'_gtFine_labelIds.png'
            # print(name)
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, name))
            self.files.append({
                "img": img_file,
                "label":label_file,
                "name": name
            })

        self.id_to_trainid = self.info['label2train']
            


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        
        image = Image.open(datafiles["img"]).convert('RGB')
        tem = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        tem = tem.resize(self.crop_size, Image.NEAREST)
        
        image = np.asarray(image, np.float32)
        tem = np.asarray(tem, np.float32)
        label = np.copy(tem)

        for x, y in self.id_to_trainid:
            label[tem==x] = y

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name

if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
