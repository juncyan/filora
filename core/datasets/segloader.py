import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from skimage import io
import imageio
import paddle
from paddle.io import Dataset
import core.datasets.imutils as imutils

class SegReader(Dataset):
    def __init__(self,path_root="./dataset/",mode="train"):
        super(SegReader,self).__init__()

        self.aug = mode == "train"
        self.path_root = os.path.join(path_root, mode)
        self.data_list = self._get_list(self.path_root)
        self.data_num = len(self.data_list)
        if os.path.exists(os.path.join(path_root, 'label_info.csv')):
            self.label_info = pd.read_csv(os.path.join(path_root, 'label_info.csv'))
        else:
            self.label_info = np.array([[0, 0, 0], [255, 255, 255]])


        self.file_name = []
        self.sst1_images = []
        self.sst1_lab = []
        
        for _file in self.data_list:
            self.sst1_images.append(os.path.join(self.path_root, "image", _file))
            self.sst1_lab.append(os.path.join(self.path_root, "label", _file))
            self.file_name.append(_file)
               

    def __getitem__(self, index):

        A_path = self.sst1_images[index]
        labA_path = self.sst1_lab[index]
        sst1 = np.array(Image.open(A_path).convert('RGB'), np.float32)
        label = np.array(imageio.imread(labA_path))

        if len(label.shape) == 2:
            label = np.array(label, np.int64)
        elif len(label.shape) == 3:
            label = imutils.one_hot_it(label, self.label_info)
            label = np.argmax(label, axis=-1)
        else:
            raise ValueError("label shape error")
        
        sst1, label = self.__transforms(self.aug, sst1, label)

        sst1 = paddle.to_tensor(sst1)
        label = paddle.to_tensor(label.astype(np.int64))
        
        if self.aug:
            return sst1, label
        return sst1, label, self.file_name[index]

    def __len__(self):
        return self.data_num

    def _get_list(self, list_path):
        data_list = os.listdir(os.path.join(list_path,'image'))
        return data_list
    
    def __transforms(self, aug, img, label):
        if aug:
            img, label = imutils.random_fliplr_seg(img, label)
            img, label = imutils.random_flipud_seg(img, label)
            img, label = imutils.random_rot_seg(img, label)

        img = imutils.normalize_img(img)  # imagenet normalization
        img = np.transpose(img, (2, 0, 1))
        return img, label


if __name__ == "__main__":
    dataset_path = '/mnt/data/Datasets/CLCD'
    x = np.random.random([4,4,3])
    mean = np.std(x, axis=(0,1))
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    print(x)
    print(mean)

    