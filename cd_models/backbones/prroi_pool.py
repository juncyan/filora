import paddle
from paddle.autograd import PyLayer
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

class PrroiPool(nn.Layer):
    def __init__(self, out_size, spatial=1.0, batch_roi_nums=None):
        super().__init__()
        self.pooled_height, self.pooled_width = out_size
        self.spatial = spatial
        self.batch_roi_nums = batch_roi_nums
        
    
    def forward(self, x, rois):
        N = x.shape[0]
        batch_roi_nums = self.batch_roi_nums
        if self.batch_roi_nums == None:
            batch_roi_nums = (1,) * N
        res = []
        for i in range(N):
            d = x[i]
            roi = np.array(rois[i])* self.spatial
            roi = np.round(roi)
            roi_width = int(roi[2] - roi[0] +1)
            roi_height = int(roi[3] - roi[1] + 1)
            bin_size_w = roi_width / self.pooled_width
            bin_size_h = roi_height / self.pooled_height
            init = paddle.sum(d[:, roi[0]: roi[2], roi[1]:roi[3]], axis=0)
            r = init / roi_width * roi_height
            res.append(r)
        return paddle.concat(res)

class SPP(nn.Layer):
    def __init__(self, levels=[4,2,1]):
        super().__init__()
        self.levels = levels
        
    def forward(self, x):
        bs, c, h, w = x.shape
        y = []
        for level in self.levels:
            kh = int(h/level)
            kw = int(w/level)
            for i in range(level):
                for j in range(level):
                    pool = x[:, :, i*kh:(i+1)*kh, j*kw:(j+1)*kw]
                    t = F.adaptive_max_pool2d(pool, 1)
                    t = paddle.reshape(t, [bs, -1])
                    y.append(t)
        y = paddle.concat(y, axis=1)
     
        return y