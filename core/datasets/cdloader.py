import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import paddle
from paddle.io import Dataset
from .utils import one_hot_it
from skimage import io
import imageio
import core.datasets.imutils as imutils

class CDReader(Dataset):
    def __init__(self,path_root="./dataset/",mode="train"):
        super(CDReader,self).__init__()

        self.aug = mode == "train"
        self.path_root = os.path.join(path_root, mode)
        self.data_list = self._get_list(self.path_root)
        self.data_num = len(self.data_list)

        self.label_info = pd.read_csv(os.path.join(path_root, 'label_info.csv'))
        self.label_color = np.array([[1, 0], [0, 1]])

        self.file_name = []
        self.sst1_images = []
        self.sst2_images = []
        self.gt_images = []
        
        for _file in self.data_list:
            self.sst1_images.append(os.path.join(self.path_root, "A", _file))
            self.sst2_images.append(os.path.join(self.path_root, "B", _file))
            self.gt_images.append(os.path.join(self.path_root, "label", _file))
            self.file_name.append(_file)
               

    def __getitem__(self, index):

        A_path = self.sst1_images[index]
        B_path = self.sst2_images[index]
        cd_path = self.gt_images[index]

        A_img = np.array(imageio.imread(A_path, pilmode="RGB"), np.float32)
        B_img = np.array(imageio.imread(B_path, pilmode="RGB"), np.float32)
      
        label = np.array(imageio.imread(cd_path), np.float16)

        sst1, sst2, label = self.__transforms(self.aug, A_img, B_img, label)

        if (len(label.shape) == 3):
            label = one_hot_it(label, self.label_info)
        elif (len(label.shape) == 2):
            label = np.array((label != 0), dtype=np.int8)
            label = self.label_color[label]

        label = np.array(np.transpose(label, [2,0,1]), dtype=np.float32)
        # label = paddle.to_tensor(label).astype('int64')
    
        sst1 = paddle.to_tensor(sst1)
        sst2 = paddle.to_tensor(sst2)
        label = paddle.to_tensor(label)
        image = paddle.concat([sst1, sst2], axis=0)

        if self.aug:
            data = {"img": image, "label": label}
        else:
            data = {"img": image, "label": label, 'name': self.file_name[index]}
        return data
        
    def __len__(self):
        return self.data_num

    def _get_list(self, list_path):
        data_list = os.listdir(os.path.join(list_path,'A'))
        return data_list
    
    def __transforms(self, aug, pre_img, post_img, cd_label):
        if aug:
            # pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, cd_label = imutils.random_fliplr(pre_img, post_img, cd_label)
            pre_img, post_img, cd_label = imutils.random_flipud(pre_img, post_img, cd_label)
            pre_img, post_img, cd_label = imutils.random_rot(pre_img, post_img, cd_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))
        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, cd_label

class SBCDReader(Dataset):
    def __init__(self,path_root="./dataset/",mode="train"):
        super(SBCDReader,self).__init__()

        self.aug = mode == "train"
        self.path_root = os.path.join(path_root, mode)
        self.data_list = self._get_list(self.path_root)
        self.data_num = len(self.data_list)
        self.label_color = np.array([[1, 0], [0, 1]])

        self.file_name = []
        self.sst1_images = []
        self.sst1_lab = []
        self.sst2_images = []
        # self.gt_images = []
        
        for _file in self.data_list:
            self.sst1_images.append(os.path.join(self.path_root, "A", _file))
            self.sst2_images.append(os.path.join(self.path_root, "B", _file))
            self.sst1_lab.append(os.path.join(self.path_root, "labelA", _file))
            # self.gt_images.append(os.path.join(self.path_root, "label", _file))
            self.file_name.append(_file)
               

    def __getitem__(self, index):

        A_path = self.sst1_images[index]
        B_path = self.sst2_images[index]
        labA_path = self.sst1_lab[index]
        # cd_path = self.gt_images[index]

        A_img = np.array(imageio.imread(A_path, pilmode="RGB"), np.float32)
        B_img = np.array(imageio.imread(B_path, pilmode="RGB"), np.float32)
        
        labA = np.array(imageio.imread(labA_path), np.float32)
    
        sst1, sst2, gt1 = self.__transforms(self.aug, A_img, B_img, labA)
    
        sst1 = paddle.to_tensor(sst1)
        sst2 = paddle.to_tensor(sst2)
        
        lab = np.array(gt1 > 0, dtype=np.uint8)
        lab = self.label_color[lab]
        lab = np.array(np.transpose(lab, [2,0,1]), dtype=np.float32)
        lab = paddle.to_tensor(lab)
        # print(sst1.shape, sst2.shape, gt1.shape, gt2.shape, gtcd.shape)
        image = paddle.concat([sst1, sst2], axis=0)

        if self.aug:
            data = {"img": image, "label": lab}
        else:
            data = {"img": image, "label": lab, 'name': self.file_name[index]}
        return data

    def __len__(self):
        return self.data_num

    def _get_list(self, list_path):
        data_list = os.listdir(os.path.join(list_path,'A'))
        return data_list
    
    def __transforms(self, aug, pre_img, post_img, cd_label):
        if aug:
            # pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, cd_label = imutils.random_fliplr(pre_img, post_img, cd_label)
            pre_img, post_img, cd_label = imutils.random_flipud(pre_img, post_img, cd_label)
            pre_img, post_img, cd_label = imutils.random_rot(pre_img, post_img, cd_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))
        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, cd_label

class DataReader(Dataset):
    # 这里的transforms、num_classes和ignore_index需要，避免PaddleSeg在Eval时报错
    def __init__(self, dataset_path, mode, load_edge = False, en_concat = True):
        
        self.data_dir = os.path.join(dataset_path, mode) #dataset_path

        self.load_edge = load_edge
        self.en_concat = en_concat

        self.data_list = self._get_list(self.data_dir)
        self.data_num = len(self.data_list)

        self.label_info = pd.read_csv(os.path.join(dataset_path, 'label_info.csv'))
        self.label_color = np.array([[1, 0], [0, 1]])


        assert self.data_list != None, "no data list could load"

        self.sst1_images = []
        self.sst1_edge = []
        self.sst2_images = []
        self.sst2_edge = []
        self.gt_images = []
        
        if self.load_edge:
            for _file in self.data_list:
                self.sst1_images.append(os.path.join(f"{self.data_dir}/A", _file))
                self.sst2_images.append(os.path.join(f"{self.data_dir}/B", _file))
                self.sst1_edge.append(os.path.join(f"{self.data_dir}/AEdge", _file))
                self.sst2_edge.append(os.path.join(f"{self.data_dir}/BEdge", _file))
                self.gt_images.append(os.path.join(f"{self.data_dir}/label", _file))
                
        else:
            for _file in self.data_list:
                self.sst1_images.append(os.path.join(f"{self.data_dir}/A", _file))
                self.sst2_images.append(os.path.join(f"{self.data_dir}/B", _file))
                self.gt_images.append(os.path.join(f"{self.data_dir}/label", _file))
                

    def __getitem__(self, index):
        # print(self.data_list[index])
        A_path = self.sst1_images[index]
        B_path = self.sst2_images[index]
        lab_path = self.gt_images[index]

        A_img = self._normalize(np.array(Image.open(A_path)))
        B_img = self._normalize(np.array(Image.open(B_path)))

        if self.load_edge:
            AEdge_path = self.sst1_edge[index]
            BEdge_path = self.sst2_edge[index]
            edge1 = cv2.imread(AEdge_path, cv2.IMREAD_UNCHANGED)
            edge2 = cv2.imread(BEdge_path, cv2.IMREAD_UNCHANGED)
            A_img = np.concatenate((A_img, edge1[..., np.newaxis]), axis=-1)  # 将两个时段的数据concat在通道层
            B_img = np.concatenate((B_img, edge2[..., np.newaxis]), axis=-1)  # 将两个时段的数据concat在通道层
        # w, h, _ = A_img.shape

        A_img = paddle.to_tensor(A_img.transpose(2, 0, 1)).astype('float32')
        B_img = paddle.to_tensor(B_img.transpose(2, 0, 1)).astype('float32')

        label = np.array(Image.open(lab_path))  # cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)

        if (len(label.shape) == 3):
            label = one_hot_it(label, self.label_info)
        elif (len(label.shape) == 2):
            label = np.array((label != 0), dtype=np.int8)
            label = self.label_color[label]

        # label = np.argmax(label, axis=-1)
        # label = paddle.to_tensor(label[np.newaxis, :]).astype('int64')
        label = np.transpose(label, [2,0,1])
        label = paddle.to_tensor(label).astype('int64')

        if self.en_concat:
            image = paddle.concat([A_img, B_img], axis=0)
            data = {"img": image, "label": label}
            return data
        else:
            data = {"img1": A_img, "img2":B_img, "label": label}
            return data

    def __len__(self):
        return self.data_num

    # 这个用于把list.txt读取并转为list
    def _get_list(self, list_path):
        # data_list = None
        # with open(list_path, 'r') as f:
        #     data_list = f.read().split('\n')[:-1]
        # return data_list
        data_list = os.listdir(os.path.join(list_path,'A'))
        return data_list

    @staticmethod
    def _normalize(img, mean=[0.485, 0.456, 0.406], std=[1, 1, 1]):
        im = img.astype(np.float32, copy=False)
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        im = im / 255.0
        im -= mean
        im /= std
        return im


class TestReader(DataReader):
    def __init__(self, dataset_path, mode = "test", load_edge = False, en_concat = True):
        super(TestReader, self).__init__(dataset_path, mode, load_edge, en_concat)
        # data_dir = os.path.join(dataset_path, 'test')
        # self.data_list = self._get_list(self.data_dir)
        #
        # self.data_num = len(self.data_list)
        #
        # self.label_info = pd.read_csv(os.path.join(dataset_path, 'label_info.csv'))
        self.data_name = os.path.split(dataset_path)[-1]
        
        # self.sst1_images = []
        # self.sst1_edge = []
        # self.sst2_images = []
        # self.sst2_edge = []
        # self.gt_images = []
        self.file_name = []
        
        if self.load_edge:
            for _file in self.data_list:
                self.sst1_images.append(os.path.join(f"{self.data_dir}/A", _file))
                self.sst2_images.append(os.path.join(f"{self.data_dir}/B", _file))
                self.sst1_edge.append(os.path.join(f"{self.data_dir}/AEdge", _file))
                self.sst2_edge.append(os.path.join(f"{self.data_dir}/BEdge", _file))
                self.gt_images.append(os.path.join(f"{self.data_dir}/label", _file))
                self.file_name.append(_file)
        else:
            for _file in self.data_list:
                self.sst1_images.append(os.path.join(f"{self.data_dir}/A", _file))
                self.sst2_images.append(os.path.join(f"{self.data_dir}/B", _file))
                self.gt_images.append(os.path.join(f"{self.data_dir}/label", _file))
                self.file_name.append(_file)

    def __getitem__(self, index):
        # print(self.data_list[index])
        A_path = self.sst1_images[index]
        B_path = self.sst2_images[index]
        lab_path = self.gt_images[index]

        A_img = self._normalize(np.array(Image.open(A_path)))
        B_img = self._normalize(np.array(Image.open(B_path)))

        if self.load_edge:
            AEdge_path = self.sst1_edge[index]
            BEdge_path = self.sst2_edge[index]
            edge1 = cv2.imread(AEdge_path, cv2.IMREAD_UNCHANGED)
            edge2 = cv2.imread(BEdge_path, cv2.IMREAD_UNCHANGED)
            A_img = np.concatenate((A_img, edge1[..., np.newaxis]), axis=-1)  # 将两个时段的数据concat在通道层
            B_img = np.concatenate((B_img, edge2[..., np.newaxis]), axis=-1)  # 将两个时段的数据concat在通道层
        # w, h, _ = A_img.shape

        A_img = paddle.to_tensor(A_img.transpose(2, 0, 1)).astype('float32')
        B_img = paddle.to_tensor(B_img.transpose(2, 0, 1)).astype('float32')

        label = np.array(Image.open(lab_path))  # cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)

        if (len(label.shape) == 3):
            label = one_hot_it(label, self.label_info)
        elif (len(label.shape) == 2):
            label = np.array((label != 0), dtype=np.int8)
            label = self.label_color[label]

        # label = np.argmax(label, axis=-1)
        # label = paddle.to_tensor(label[np.newaxis, :]).astype('int64')
        label = np.transpose(label, [2,0,1])
        label = paddle.to_tensor(label).astype('int64')

        if self.en_concat:
            image = paddle.concat([A_img, B_img], axis=0)
            data = {"img": image, "label": label, 'name': self.file_name[index]}
            return data
        else:
            data = {"img1": A_img, "img2":B_img, "label": label, 'name': self.file_name[index]}
            return data


def detect_building_edge(data_path, save_pic_path):
    canny_low = 180
    canny_high = 210
    hough_threshold = 64
    hough_minLineLength = 16
    hough_maxLineGap = 3
    hough_rho = 1
    hough_theta = np.pi / 180
    image_names=os.listdir(data_path)
    for image_name in image_names:
        img=cv2.imread(os.path.join(data_path, image_name))
        shape=img.shape[:2]
        img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        edges=cv2.Canny(img_gray,canny_low,canny_high)
        lines=cv2.HoughLinesP(edges,hough_rho,hough_theta,hough_threshold,hough_minLineLength,hough_maxLineGap)
        line_pic = np.zeros(shape, np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_pic, (x1, y1), (x2, y2), 1, thickness=1)
        cv2.imwrite(os.path.join(save_pic_path, image_name),line_pic)


if __name__ == "__main__":
    dataset_path = '/mnt/data/Datasets/CLCD'
    x = np.random.random([4,4,3])
    mean = np.std(x, axis=(0,1))
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    print(x)
    print(mean)
