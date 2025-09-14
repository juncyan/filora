import paddle
from paddle import nn


class BCL(nn.Layer):  
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """
    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label[label == 1] = -1
        label[label == 0] = 1

        mask = paddle.to_tensor((label != 255), dtype=paddle.float16)
        distance = distance * mask

        pos_num = paddle.sum(paddle.to_tensor((label == 1), dtype=paddle.float16))+0.0001
        neg_num = paddle.sum(paddle.to_tensor((label == -1), dtype=paddle.float16))+0.0001

        loss_1 = paddle.sum((1+label) / 2 * paddle.pow(distance, 2)) /pos_num
        loss_2 = paddle.sum((1-label) / 2 *
            paddle.pow(paddle.clip(self.margin - distance, min=0.0), 2)
        ) / neg_num
        # paddle.clip()
        loss = loss_1 + loss_2
        return loss