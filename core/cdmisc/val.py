# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os

import numpy as np
import time
import paddle
import paddle.nn.functional as F
import pandas as pd
from tqdm import tqdm
from paddleseg.utils import TimeAverager
from .metrics import Metrics

np.set_printoptions(suppress=True)


def evaluate(model, val_dataset, args=None):
    """
    Launch evalution.
    """
    assert args != None, "args is None, please check!"
    
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    evaluator = Metrics(num_class=args.num_classes)
    model.eval()

    with paddle.no_grad():
        for data in tqdm(val_dataset):
            reader_cost_averager.record(time.time() - batch_start)

            label = data['label'].astype('int64')
            
            images = data['img'].cuda()
            pred = model(images)
            
            if hasattr(model, "predict"):
                pred = model.predict(pred)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[0]

            # if pred.shape[1] > 1:
            #     pred = paddle.argmax(pred, axis=1)
            # pred = pred.squeeze()

            # if label.shape[1] > 1:
            #     label = paddle.argmax(label, 1)
            # label = label.squeeze()

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

            evaluator.add_batch(pred.cpu(), label)

    metrics = evaluator.Get_Metric()
    pa = metrics["pa"]
    miou = metrics["miou"]
    mf1 = metrics["mf1"]
    kappa = metrics["kappa"]
    iou1 = metrics["iou_1"]

    if args.logger != None:
        infor = "[EVAL] Images: {} batch_cost {:.4f}, reader_cost {:.4f}".format(args.val_num, batch_cost, reader_cost)
        args.logger.info(infor)
        args.logger.info("[METRICS] PA:{:.4},mIoU:{:.4},kappa:{:.4},Macro_f1:{:.4}, IoU 1:{:.4f}".format(pa,miou,kappa,mf1, iou1))
        
    
    d = pd.DataFrame([metrics])
    if os.path.exists(args.metric_path):
        d.to_csv(args.metric_path,mode='a', index=False, header=False,float_format="%.4f")
    else:
        d.to_csv(args.metric_path, index=False,float_format="%.4f")
        
    return iou1
