# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


from .layers import make_norm, Conv3x3, CBAM
from .stanet import Backbone, Decoder
from .losses.fccdn_loss import DiceLoss
from .losses.bcl_loss import BCL


class DSAMNet(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an 
        Open Aerial Image Dataset for Remote Sensing Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based 
        head to stablize the training process.

    Args:
        in_channels (int): Number of bands of the input images.
        num_classes (int): Number of target classes.
        ca_ratio (int, optional): Channel reduction ratio for the channel 
            attention module. Default: 8.
        sa_kernel (int, optional): Size of the convolutional kernel used in the 
            spatial attention module. Default: 7.
    """

    def __init__(self, in_channels, num_classes, ca_ratio=8, sa_kernel=7):
        super(DSAMNet, self).__init__()

        WIDTH = 64

        self.backbone = Backbone(
            in_ch=in_channels, arch='resnet18', strides=(1, 1, 2, 2, 1))
        self.decoder = Decoder(WIDTH)

        self.cbam1 = CBAM(64, ratio=ca_ratio, kernel_size=sa_kernel)
        self.cbam2 = CBAM(64, ratio=ca_ratio, kernel_size=sa_kernel)

        self.dsl2 = DSLayer(64, num_classes, 32, stride=2, output_padding=1)
        self.dsl3 = DSLayer(128, num_classes, 32, stride=4, output_padding=3)

        # self.conv_out = nn.Sequential(
        #     Conv3x3(
        #         WIDTH, WIDTH, norm=True, act=True),
        #     Conv3x3(WIDTH, num_classes))

        self.init_weight()

    def forward(self, x):
        t1, t2 = x[:, :3, :, :], x[:, 3:, :, :]
        f1 = self.backbone(t1)
        f2 = self.backbone(t2)

        y1 = self.decoder(f1)
        y2 = self.decoder(f2)

        y1 = self.cbam1(y1).transpose([0,3,2,1])
        y2 = self.cbam2(y2).transpose([0,3,2,1])
        out = F.pairwise_distance(y1,y2,keepdim=True).transpose([0,3,2,1])
        # out = paddle.abs(y1 - y2)
        out = F.interpolate(
            out, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        pred = out

        if not self.training:
            return [pred]
        else:
            ds2 = self.dsl2(paddle.abs(f1[0] - f2[0]))
            ds3 = self.dsl3(paddle.abs(f1[1] - f2[1]))
            return [pred, ds2, ds3]

    def init_weight(self):
        pass

    @staticmethod
    def loss(pred, label, wdice=0.2):
        # label = paddle.argmax(label,axis=1)
        prob, ds2, ds3 = pred
        dsloss2 = DiceLoss()(ds2, label)
        dsloss3 = DiceLoss()(ds3, label)
        Dice_loss = 0.5*(dsloss2+dsloss3)

        label = paddle.argmax(label, 1).unsqueeze(1)
        label = paddle.to_tensor(label, paddle.float16)
        # print(prob.shape, label.shape)
        CT_loss = BCL()(prob, label)
        CD_loss = CT_loss + wdice * Dice_loss
        return CD_loss
    
    @staticmethod
    def predict(pred):
        prob = pred[0]
        prob = paddle.to_tensor((prob > 1),paddle.int16)
        return prob


class DSLayer(nn.Sequential):
    def __init__(self, in_ch, out_ch, itm_ch, **convd_kwargs):
        super(DSLayer, self).__init__(
            nn.Conv2DTranspose(
                in_ch, itm_ch, kernel_size=3, padding=1, **convd_kwargs),
            make_norm(itm_ch),
            nn.ReLU(),
            nn.Dropout2D(p=0.2),
            nn.Conv2DTranspose(
                itm_ch, out_ch, kernel_size=3, padding=1))



# class DiceLoss(nn.Layer):
#     def __init__(self, num_class=2):
#         super().__init__()
#         self.__num_class = num_class
    
#     def forward(self, pred, lab):
#         if len(lab.shape) == 4 and lab.shape[1] != 1:
#             lab = paddle.argmax(lab, axis=1)

#         if len(pred.shape) == 4 and pred.shape[1] != 1:
#             pred = paddle.argmax(pred, axis=1)

        
#         gt_image = paddle.squeeze(lab)
#         pre_image = paddle.squeeze(pred)
        
#         assert (len(gt_image) == len(pre_image))
#         assert gt_image.shape == pre_image.shape

#         mask = (gt_image >= 0) & (gt_image < self.__num_class)
#         label = self.__num_class * gt_image[mask].astype('int') + pre_image[mask]
#         count = paddle.bincount(label, minlength=self.__num_class ** 2)
        
#         confusion_matrix = count.reshape([self.__num_class, self.__num_class])
#         dice = 2 * paddle.diag(confusion_matrix) / (1e-5+paddle.sum(confusion_matrix, axis=0) + paddle.sum(confusion_matrix, axis=1))

#         return 1. - paddle.nanmean(dice)


  
# class BCL(nn.Layer):  
#     """
#     batch-balanced contrastive loss
#     no-change，1
#     change，-1
#     """
#     def __init__(self, margin=2.0):
#         super(BCL, self).__init__()
#         self.margin = margin

#     def forward(self, distance, label):
#         label[label == 1] = -1
#         label[label == 0] = 1

#         mask = (label != 255).float()
#         distance = distance * mask

#         pos_num = paddle.sum((label==1).float())+0.0001
#         neg_num = paddle.sum((label==-1).float())+0.0001

#         loss_1 = paddle.sum((1+label) / 2 * paddle.pow(distance, 2)) /pos_num
#         loss_2 = paddle.sum((1-label) / 2 *
#             paddle.pow(paddle.clamp(self.margin - distance, min=0.0), 2)
#         ) / neg_num
#         loss = loss_1 + loss_2
#         return loss

if __name__ == "__main__":
    print("TNet.Metrics run")
    x = [0.83611815, 0.51126306, 0.69839933, 0.75191475]
    print(np.mean(x))






