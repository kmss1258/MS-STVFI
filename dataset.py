import os
import cv2
import ast
import io
import torch
import numpy as np
import random
from script.resize import imresize_np
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdobeDataset(Dataset):
    def __init__(self, dataset_name, root_dir, batch_size=32):
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.load_data()
        self.allframe = True

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        def read(name):
            data_list = []
            with open(os.path.join(self.root_dir, f"adobe240fps_folder_{name}.txt")) as f:
                data = f.readlines()
                for l in data:
                    l = l.strip('\n')
                    path = os.path.join(self.root_dir, "images", f'{l}')
                    interval = 1
                    if name != 'train':
                        interval = 9
                    for i in range(0, len(os.listdir(path)) - 9, interval):
                        data_tuple = []
                        for j in range(9):
                            data_tuple.append('{}/{:05d}.png'.format(path, i + j))
                        data_list.append(data_tuple)
            return data_list

        self.meta_data = read(self.dataset_name)
        self.nr_sample = len(self.meta_data)

    def aug(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x + h, y:y + w, :]
        return imgs

    def read(self, x):
        return cv2.imread(x)

    def getimg(self, index, training=False):
        data = self.meta_data[index]
        if not training:
            imgs = []
            if self.allframe:
                for i in range(9):
                    imgs.append(self.read(data[i]))
            else:
                imgs.append(self.read(data[0]))
                imgs.append(self.read(data[4]))
                imgs.append(self.read(data[8]))
            step = 0.5
        else:
            ind = [1, 2, 3, 4, 5, 6, 7]
            random.shuffle(ind)
            ind[1] = ind[0]
            ind[0] = 0
            ind[2] = 8
            img0 = self.read(data[ind[0]])
            gt = self.read(data[ind[1]])
            img1 = self.read(data[ind[2]])
            step = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0])
            imgs = [img0, gt, img1]
        return imgs, step

    def __getitem__(self, index):
        if self.dataset_name == 'train':
            imgs, timestep = self.getimg(index, training=True)
            imgs = self.aug(imgs, 128, 128)
            img0, gt, img1 = imgs
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep
            imgs = [img0.copy(), gt.copy(), img1.copy()]
        else:
            imgs, timestep = self.getimg(index, training=False)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        lowres = []
        for img in imgs:
            lowres.append(cv2.resize(imresize_np(img, 0.25), (0, 0), fx=4, fy=4,
                                     interpolation=cv2.INTER_AREA))  # before cv2.INTER_CUBIC after cv2.INTER_AREA
        imgs = torch.from_numpy(np.concatenate(imgs.copy(), 2)).permute(2, 0, 1)
        lowres = torch.from_numpy(np.concatenate(lowres.copy(), 2)).permute(2, 0, 1)
        return imgs, lowres, timestep


if __name__ == '__main__':
    ds = DataLoader(AdobeDataset('train', root_dir="/media/ms-neo2/ms-ssd11/1.dataset/VFI/adobe240fps"))
