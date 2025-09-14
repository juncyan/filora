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
import time
from collections import deque
import shutil
import logging
from copy import deepcopy

import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.optimizer
from tqdm import tqdm
from .loss import loss_lovasz, CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity
from .val import evaluate
from .predict import test

def train(model, train_loader, val_loader, test_loader, args):
    """
    Launch training.

    Args:
        model（nn.Layer): A semantic segmentation model.
        train_dataset (paddle.io.Dataset): Used to read and process training datasets.
        val_dataset (paddle.io.Dataset, optional): Used to read and process validation datasets.
        optimizer (paddle.optimizer.Optimizer): The optimizer.
    """
    # paddle.optimizer.lr.PolynomialDecay(args.lr, args.iters)
    lr = paddle.optimizer.lr.CosineAnnealingDecay(args.lr, T_max=(args.iters // 3), last_epoch=0.5)  # 余弦衰减
    optimizer = paddle.optimizer.AdamW(lr, parameters=model.parameters()) 

    model.train()
    if args.logger != None:
        args.logger.info("start train")

    best_mean_iou = -1.0
    best_model_iter = -1

    batch_start = time.time()

    criterion = CrossEntropyLoss2d(ignore_index=0)
    criterion_sc = ChangeSimilarity()

    for _epoch in range(args.iters):
        avg_loss_list = []
        epoch = _epoch + 1
        model.train()

        for img1, img2, gt1, gt2, gt in tqdm(train_loader):
            # paddle.Tensor().type(paddle.float32)
            img1 = img1.cuda()
            img2 = img2.cuda()
            gt1 = gt1.cuda()
            gt2 = gt2.cuda()
            gt = gt.cuda()

            cd, sem1, sem2 = model(img1, img2)
            # loss = loss_lovasz(cd, sem1, sem2, gt1, gt2, gt)  
            # print(gt.shape, sem1.shape, sem2.shape, gt1.shape, gt2.shape)
            loss_seg = criterion(sem1, gt1) * 0.5 +  criterion(sem2, gt2) * 0.5  
            loss_bn = weighted_BCE_logits(cd, gt)
            loss_sc = criterion_sc(sem1[:,1:], sem2[:,1:], gt)
            loss = loss_seg + loss_bn + loss_sc
            
            loss.backward()
            optimizer.step()
            lr = optimizer.get_lr()

            # update lr
            lr_sche = optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                if isinstance(lr_sche, paddle.optimizer.lr.ReduceOnPlateau):
                    lr_sche.step(loss)
                else:
                    lr_sche.step()

            model.clear_gradients()
            
            avg_loss = np.array(loss.cpu())
            avg_loss_list.append(avg_loss)

        batch_cost_averager = time.time() - batch_start
        avg_loss = np.mean(avg_loss_list)


        if args.logger != None:
            args.logger.info(
                "[TRAIN] iter: {}/{}, loss: {:.4f}, lr: {:.6}, batch_cost: {:.2f}, ips: {:.4f} samples/sec".format(
                    epoch, args.iters, avg_loss, lr, batch_cost_averager, batch_cost_averager / args.traindata_num))

        if epoch == args.iters:
            paddle.save(model.state_dict(),
                        os.path.join(args.save_dir, f'last_model.pdparams'))
        mean_iou = evaluate(model, val_loader, args)

        if mean_iou > best_mean_iou:
            # predict(model, test_data_loader, args)
            best_mean_iou = mean_iou
            best_model_iter = epoch
            paddle.save(model.state_dict(), args.best_model_path)

        if args.logger !=  None:
            args.logger.info("[TRAIN] best iter {}, max mIoU {:.4f}".format(best_model_iter, best_mean_iou))
        batch_start = time.time()
    
    test(model, test_loader, args)
    # test(model, test_loader, args)
    logging.shutdown()
    
    # Sleep for a second to let dataloader release resources.
    time.sleep(10)
