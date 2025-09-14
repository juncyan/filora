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
import paddle.tensor
import pandas as pd
from tqdm import tqdm
from paddleseg.utils import TimeAverager
from .metric import Metric_SCD

np.set_printoptions(suppress=True)


def evaluate(model, val_loader, args):
    """
    Launch evalution.
    """
    assert args != None, "args is None, please check!"
    
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    model.eval()
    evaluator = Metric_SCD(num_class=args.num_classes)

    with paddle.no_grad():
        for img1, img2, gt1, gt2, gt,_ in tqdm(val_loader):
            reader_cost_averager.record(time.time() - batch_start)

            img1 = img1.cuda()
            img2 = img2.cuda()
            cd, sem1, sem2 = model(img1, img2)
                 
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(cd))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

            change_mask = F.sigmoid(cd).cpu().detach()>0.5 #paddle.argmax(cd, axis=1)
            change_mask = change_mask.squeeze()
            change_mask = change_mask.cast('int64')
            sem1 = paddle.argmax(sem1, axis=1)
            sem2 = paddle.argmax(sem2, axis=1)

            sem1 = (sem1*change_mask).cpu().numpy()
            sem2 = (sem2*change_mask).cpu().numpy()
            
            evaluator.add_batch(sem1, gt1)
            evaluator.add_batch(sem2, gt2)

    metrics = evaluator.Get_Metric()
    evaluator.reset()

    miou = metrics['miou']

    if args.logger != None:
        infor = "[EVAL] Images: {} batch_cost {:.4f}, reader_cost {:.4f}".format(args.val_num, batch_cost, reader_cost)
        args.logger.info(infor)
        args.logger.info("[METRICS] MIoU:{:.4}, Kappa:{:.4}, F1:{:.4}, Sek:{:.4}".format(
            miou,metrics['kappa'],metrics['f1'],metrics['sek']))
        args.logger.info("[METRICS] PA:{:.4}, Prec.:{:.4}, Recall:{:.4}".format(
            metrics['pa'],metrics['prec'],metrics['recall']))
        
    
    d = pd.DataFrame([metrics])
    if os.path.exists(args.metric_path):
        d.to_csv(args.metric_path,mode='a', index=False, header=False,float_format="%.4f")
    else:
        d.to_csv(args.metric_path, index=False,float_format="%.4f")
        
    return miou
