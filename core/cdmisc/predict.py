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
import cv2
import numpy as np
import time
import paddle
import datetime
import glob
import pandas as pd

from paddleseg.utils import TimeAverager, op_flops_funs
from .metrics import Metrics
from .logger import load_logger
from .count_params import flops


def predict(model, dataset, weight_path=None, data_name="test", num_classes=2):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A semantic segmentation model.
        dataset (paddle.io.DataLoader): Used to read and process test datasets.
        weights_path (string, optional): weights saved local.
    """

    model.eval()
    if weight_path:
        layer_state_dict = paddle.load(f"{weight_path}")
        model.set_state_dict(layer_state_dict)
    else:
        exit()

    img_ab_concat = True

    time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")
    model_name = model.__str__().split("(")[0]

    img_dir = f"/mnt/data/Results/{data_name}/{model_name}_{time_flag}"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    color_label = np.array([[0,0,0],[255,255,255],[0,128,0],[0,0,128]])

    logger = load_logger(f"{img_dir}/prediction.log")
    logger.info(f"test {model_name} on {data_name} save in {img_dir}")

    batch_sampler = paddle.io.BatchSampler(
        dataset, batch_size=8, shuffle=False, drop_last=False)
    
    loader = paddle.io.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        return_list=True)

    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    evaluator = Metrics(num_class=2)
    model.eval()
    with paddle.no_grad():
        for data in loader:

            reader_cost_averager.record(time.time() - batch_start)

            name = data['name']
            label = data['label'].astype('int64')

            images = data['img'].cuda()
            pred = model(images)
                
            if hasattr(model, "predict"):
                pred = model.predict(pred)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[0]

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

            pred = paddle.argmax(pred, axis=1)
            pred = pred.squeeze().cpu()

            if label.shape[1] > 1:
                label = paddle.argmax(label, 1)
            label = label.squeeze()
            label = np.array(label)

            evaluator.add_batch(pred, label)
            
            for idx, ipred in enumerate(pred):
                ipred = ipred.numpy()
                if (np.max(ipred) != np.min(ipred)):
                    flag = (label[idx] - ipred)
                    ipred[flag == -1] = 2
                    ipred[flag == 1] = 3
                    img = color_label[ipred]
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(f"{img_dir}/{name[idx]}", img)

    evaluator.calc()
    miou = evaluator.Mean_Intersection_over_Union()
    acc = evaluator.Pixel_Accuracy()
    class_iou = evaluator.Intersection_over_Union()
    class_precision = evaluator.Class_Precision()
    kappa = evaluator.Kappa()
    # m_dice = evaluator.Mean_Dice()
    f1 = evaluator.F1_score()
    macro_f1 = evaluator.Macro_F1()
    class_recall = evaluator.Recall()

    infor = "[PREDICT] #Images: {} batch_cost {:.4f}, reader_cost {:.4f}".format(len(dataset), batch_cost, reader_cost)
    logger.info(infor)
    infor = "[METRICS] mIoU: {:.4f}, Acc: {:.4f}, Kappa: {:.4f}, Macro_F1: {:.4f}".format(
            miou, acc, kappa, macro_f1)
    logger.info(infor)

    logger.info("[METRICS] Class IoU: " + str(np.round(class_iou, 4)))
    logger.info("[METRICS] Class Precision: " + str(np.round(class_precision, 4)))
    logger.info("[METRICS] Class Recall: " + str(np.round(class_recall, 4)))
    logger.info("[METRICS] Class F1: " + str(np.round(f1, 4)))
    
    if img_ab_concat:
        images = data['img'].cuda()
        _, c, h, w = images.shape
        flop_p = flops(
        model, [1, c, h, w],
        custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

        # inputs = paddle.static.InputSpec([None,c, h, w],dtype="float16",name="inputs")
        # m = paddle.Model(model, inputs)
        # z = m.summary()
        # total_params = sum(p.numel() for p in model.parameters())
        # train_params = sum(p.numel() for p in model.parameters() if p.stop_gradient == False)
            
    else:
        img1 = data['img1'].cuda()
        _, c, h, w = img1.shape
        flop_p = flops(
        model, [1, c, h, w], 2,
        custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})
    logger.info(r"[PREDICT] model total flops is: {}, params is {}".format(flop_p["total_ops"],flop_p["total_params"]))       


def test(model, test_dataset, args=None):
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

    img_ab_concat = args.img_ab_concat

    time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")

    img_dir = f"/mnt/data/Results/{args.dataset}/{args.model}_{time_flag}"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    color_label = np.array([[0,0,0],[255,255,255],[0,128,0],[0,0,128]])

    logger = load_logger(f"{img_dir}/prediction.log")
    logger.info(f"test {args.dataset} on {args.model}")

    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    evaluator = Metrics(num_class=2)
    model.eval()
    with paddle.no_grad():
        for data in test_dataset:

            reader_cost_averager.record(time.time() - batch_start)

            name = data['name']
            label = data['label'].astype('int64')

            images = data['img'].cuda()
            pred = model(images)
                
            if hasattr(model, "predict"):
                pred = model.predict(pred)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[0]

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

            pred = paddle.argmax(pred, axis=1)
            pred = pred.squeeze().cpu()

            if label.shape[1] > 1:
                label = paddle.argmax(label, 1)
            label = label.squeeze()
            label = np.array(label)

            evaluator.add_batch(pred, label)
            
            for idx, ipred in enumerate(pred):
                ipred = ipred.numpy()
                if (np.max(ipred) != np.min(ipred)):
                    flag = (label[idx] - ipred)
                    ipred[flag == -1] = 2
                    ipred[flag == 1] = 3
                    img = color_label[ipred]
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(f"{img_dir}/{name[idx]}", img)

    evaluator.calc()
    miou = evaluator.Mean_Intersection_over_Union()
    acc = evaluator.Pixel_Accuracy()
    class_iou = evaluator.Intersection_over_Union()
    class_precision = evaluator.Class_Precision()
    kappa = evaluator.Kappa()
    # m_dice = evaluator.Mean_Dice()
    f1 = evaluator.F1_score()
    macro_f1 = evaluator.Macro_F1()
    class_recall = evaluator.Recall()

    infor = "[PREDICT] #Images: {} batch_cost {:.4f}, reader_cost {:.4f}".format(args.test_num, batch_cost, reader_cost)
    args.logger.info(infor)
    infor = "[METRICS] mIoU: {:.4f}, Acc: {:.4f}, Kappa: {:.4f}, Macro_F1: {:.4f}".format(
            miou, acc, kappa, macro_f1)
    args.logger.info(infor)

    args.logger.info("[METRICS] Class IoU: " + str(np.round(class_iou, 4)))
    args.logger.info("[METRICS] Class Precision: " + str(np.round(class_precision, 4)))
    args.logger.info("[METRICS] Class Recall: " + str(np.round(class_recall, 4)))
    args.logger.info("[METRICS] Class F1: " + str(np.round(f1, 4)))

    if img_ab_concat:
        images = data['img'].cuda()
        _, c, h, w = images.shape
        flop_p = flops(
        model, [1, c, h, w],
        custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})
            
    else:
        img1 = data['img1'].cuda()
        _, c, h, w = img1.shape
        flop_p = flops(
        model, [1, c, h, w], 2,
        custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})
    logger.info(r"[PREDICT] model total flops is: {}, params is {}".format(flop_p["total_ops"],flop_p["total_params"]))    

    img_files = glob.glob(os.path.join(img_dir, '*.png'))
    data = []
    for img_path in img_files:
        img = cv2.imread(img_path)
        lab = cls_count(img)
        # lab = np.argmax(lab, -1)
        data.append(lab)
    if data != []:
        data = np.array(data)
        pd.DataFrame(data).to_csv(os.path.join(img_dir, f'{args.model}_violin.csv'), header=['TN', 'TP', 'FP', 'FN'], index=False)

def cls_count(label):
    cls_nums = []
    color_label = np.array([[0, 0, 0], [255, 255, 255], [0, 128, 0], [0, 0, 128]])
    for info in color_label:
        color = info
        # print("label:\n", label.shape,label)
        # print("color:\n", color)
        equality = np.equal(label, color)
        matrix = np.sum(equality, axis=-1)
        nums = np.sum(matrix == 3)
        cls_nums.append(nums)
    return cls_nums

