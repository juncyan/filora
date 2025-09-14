import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import numpy as np
from paddle.autograd import PyLayer

class dcn_v2(PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias, offset, mask, 
                kernel, stride, padding, dilation, deformable_group):
        batch, in_channels, height, width = x.shape
        out_channels, kernel_num, kernel_h, kernel_w = weight.shape

        assert kernel == kernel_h and kernel == kernel_w, \
            "input shape {} and kernel shape {},{} wont match".format(kernel, kernel_h, kernel_w)
        assert in_channels == kernel_num, "Input shape {} and kernel channels {} wont match".format(in_channels, kernel_num)

        conv = nn.Conv2D(in_channels, out_channels, kernel, stride, padding, dilation,
                         weight_attr=weight, bias_attr=bias)

        height_out = (height + 2 * padding - (dilation * (kernel - 1) + 1)) / stride + 1
        width_out = (width + 2 * padding - (dilation * (kernel - 1) + 1)) / stride + 1

        ones = paddle.ones([bias.shape[0], height_out, width_out], dtype=bias.dtype)
        columns = paddle.empty({in_channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.dtype)
        output = paddle.empty({batch, out_channels, height_out, width_out}, input.dtype)

        for b in range(batch):
            input_n = x[b]
            offset_n = offset[b]
            mask_n = mask[b]
            output_n = output[b]


        ctx.save_for_backward()
        return super().forward()

    @staticmethod
    def backward(ctx, *args):
        ctx.saved_tensor()
        pass
        return super().backward(ctx, *args)
    


if __name__ == "__main__":
    print("dcn")
    x = paddle.empty([1,2,3])
    print(x)