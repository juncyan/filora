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
from skimage import io
import imageio
import datetime
from tqdm import tqdm
import glob
from paddleseg.utils import worker_init_fn
from paddleseg.utils import TimeAverager, op_flops_funs

from .metric import Metric_SCD
from core.cdmisc.logger import load_logger
from core.cdmisc.count_params import flops


def predict(model, dataset, weight_path=None, data_name="test", num_classes=2):

    model.eval()
    if weight_path:
        layer_state_dict = paddle.load(f"{weight_path}")
        model.set_state_dict(layer_state_dict)
    else:
        exit()

    time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")
    model_name = model.__str__().split("(")[0]

    img_dir = f"/mnt/data/Results/{data_name}/{model_name}_{time_flag}"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    # test_data = MusReader(self.dataset_path, datasetlist[1])
    label_info = np.transpose(dataset.label_info.values, [1,0])

    batch_sampler = paddle.io.BatchSampler(dataset, batch_size=4, shuffle=True, drop_last=True)
    test_num = dataset.__len__()
    loader = paddle.io.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=16,
            return_list=True,
            worker_init_fn=worker_init_fn, )
    
    change_color = np.array([[0,0,0],[255,255,255],[0,128,0],[0,0,128]])

    logger = load_logger(f"{img_dir}/prediction.log")
    logger.info(f"test {model_name} on {data_name}")

    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    model.eval()
    evaluator = Metric_SCD(num_class=num_classes)

    with paddle.no_grad():
        for img1, img2, gt1, gt2, gt, file in tqdm(loader):
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
            gt = np.array(gt, np.int8)
            for idx, (is1, is2, cdm) in enumerate(zip(sem1, sem2, change_mask)):
                cdm = np.array(cdm, np.uint8)
                if np.max(cdm) == np.min(cdm):
                    continue
                # flag_local = (gt[idx] - cdm)
                # cdm[flag_local == -1] = 2
                # cdm[flag_local == 1] = 3
                name = file[idx]
                # cdm = change_color[cdm]
                is1 = label_info[is1]
                is2 = label_info[is2]
                # imageio.imsave(f"{img_dir}/{name}", np.uint8(cdm))
                fa = name.replace(".", "_A.")
                fb = name.replace(".", "_B.")
                imageio.imsave(f"{img_dir}/{fa}", np.uint8(is1))
                imageio.imsave(f"{img_dir}/{fb}", np.uint8(is2))

    evaluator.get_hist(save_path=f"{img_dir}/hist.csv")

    metrics = evaluator.Get_Metric()
    evaluator.reset()

    infor = "[EVAL] Images: {} batch_cost {:.4f}, reader_cost {:.4f}".format(test_num, batch_cost, reader_cost)
    logger.info(infor)
    logger.info("[METRICS] MIoU:{:.4}, Kappa:{:.4}, F1:{:.4}, Sek:{:.4}".format(
            metrics['miou'],metrics['kappa'],metrics['f1'],metrics['sek']))
    logger.info("[METRICS] PA:{:.4}, Prec.:{:.4}, Recall:{:.4}".format(
            metrics['pa'],metrics['prec'],metrics['recall']))
    
    
    _, c, h, w = img1.shape
    flop_p = flops(
    model, [1, c, h, w], 2,
    custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})
    logger.info(r"[PREDICT] model total flops is: {}, params is {}".format(flop_p["total_ops"],flop_p["total_params"]))  


def test(model, test_loader, args):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A semantic segmentation model.
        dataset (paddle.io.DataLoader): Used to read and process test datasets.
        weights_path (string, optional): weights saved local.
    """


    assert args != None, "args is None, please check!"
    
    model.eval()
    if args.best_model_path:
        layer_state_dict = paddle.load(f"{args.best_model_path}")
        model.set_state_dict(layer_state_dict)
    else:
        exit()

    time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")

    img_dir = f"/mnt/data/Results/{args.dataset}/{args.model}_{time_flag}"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    change_color = np.array([[0,0,0],[255,255,255],[0,128,0],[0,0,128]])

    logger = load_logger(f"{img_dir}/prediction.log")
    logger.info(f"test {args.dataset} on {args.model}")

    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    model.eval()
    evaluator = Metric_SCD(num_class=args.num_classes)

    with paddle.no_grad():
        for img1, img2, gt1, gt2, gt, file in tqdm(test_loader):
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
            gt = np.array(gt, np.int8)
            if args.test:
                for idx, (is1, is2, cdm) in enumerate(zip(sem1, sem2, change_mask)):
                    cdm = np.array(cdm, np.uint8)
                    if np.max(cdm) == np.min(cdm):
                        continue
                    # flag_local = (gt[idx] - cdm)
                    # cdm[flag_local == -1] = 2
                    # cdm[flag_local == 1] = 3
                    name = file[idx]
                    # cdm = change_color[cdm]
                    is1 = args.label_info[is1]
                    is2 = args.label_info[is2]
                    # imageio.imsave(f"{img_dir}/{name}", np.uint8(cdm))
                    fa = name.replace(".", "_A.")
                    fb = name.replace(".", "_B.")
                    imageio.imsave(f"{img_dir}/{fa}", np.uint8(is1))
                    imageio.imsave(f"{img_dir}/{fb}", np.uint8(is2))

    evaluator.get_hist(save_path=f"{img_dir}/hist.csv")

    metrics = evaluator.Get_Metric()
    evaluator.reset()

    infor = "[EVAL] Images: {} batch_cost {:.4f}, reader_cost {:.4f}".format(args.test_num, batch_cost, reader_cost)
    logger.info(infor)
    logger.info("[METRICS] MIoU:{:.4}, Kappa:{:.4}, F1:{:.4}, Sek:{:.4}".format(
            metrics['miou'],metrics['kappa'],metrics['f1'],metrics['sek']))
    logger.info("[METRICS] PA:{:.4}, Prec.:{:.4}, Recall:{:.4}".format(
            metrics['pa'],metrics['prec'],metrics['recall']))
    
    
    _, c, h, w = img1.shape
    flop_p = flops(
    model, [1, c, h, w], 2,
    custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})
    logger.info(r"[PREDICT] model total flops is: {}, params is {}".format(flop_p["total_ops"],flop_p["total_params"]))  

    # img_files = glob.glob(os.path.join(img_dir, '*.png'))
    # data = []
    # for img_path in img_files:
    #     img = io.imread(img_path)
    #     lab = cls_count(img)
    #     # lab = np.argmax(lab, -1)
    #     data.append(lab)
    # if data != []:
    #     data = np.array(data)
    #     pd.DataFrame(data).to_csv(os.path.join(img_dir, f'{args.model}_violin.csv'), header=['TN', 'TP', 'FP', 'FN'], index=False)     


def cls_count(label, num_classes):
    cls_nums = []
    for k in range(num_classes):
       
        equality = np.equal(label, k)
        nums = np.sum(equality)
        # nums = np.sum(matrix == 3)
        cls_nums.append(nums)
    return cls_nums

