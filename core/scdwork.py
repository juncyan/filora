import os
import random
import paddle
import paddle.nn as nn

import numpy as np
import datetime
from paddleseg.utils import worker_init_fn
from paddleseg.models.losses import BCELoss

from .datasets import SCDReader
from .datasets.scdloader import MusReader
from .cdmisc import load_logger
from .scdmisc.train import train


class Work():
    def __init__(self, model:nn.Layer, args):
        self._seed_init()
        model_name = model.__str__().split("(")[0]
        self.args = args
        self.args.model = model_name

        paddle.device.set_device(self.args.device)
        self.model = model.to(self.args.device)

        self._seed_init()
        self.logger()
        self.dataloader()

        train(model, self.train_loader, self.val_loader, self.test_loader, self.args)

    def dataloader(self, datasetlist=['train', 'val', 'test']):
        train_data = SCDReader(self.dataset_path, datasetlist[0])
        val_data = SCDReader(self.dataset_path, datasetlist[2])
        test_data = SCDReader(self.dataset_path, datasetlist[2])

        batch_sampler = paddle.io.BatchSampler(train_data, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

        self.args.traindata_num = train_data.__len__()
        self.args.val_num = val_data.__len__()
        self.args.test_num = test_data.__len__()

        self.args.label_info = np.uint8(np.transpose(test_data.label_info.values, [1,0]))

        self.train_loader = paddle.io.DataLoader(
            train_data,
            batch_sampler=batch_sampler,
            num_workers=self.args.num_workers,
            return_list=True,
            worker_init_fn=worker_init_fn, )

        val_batch_sampler = paddle.io.BatchSampler(
            val_data, batch_size=self.args.batch_size, shuffle=False, drop_last=False)

        self.val_loader = paddle.io.DataLoader(
            val_data,
            batch_sampler=val_batch_sampler,
            num_workers=self.args.num_workers,
            return_list=True,
            worker_init_fn=worker_init_fn, )

        test_batch_sampler = paddle.io.BatchSampler(
            test_data, batch_size=self.args.batch_size, shuffle=False, drop_last=False)

        self.test_loader = paddle.io.DataLoader(
            test_data,
            batch_sampler=test_batch_sampler,
            num_workers=self.args.num_workers,
            return_list=True,
            worker_init_fn=worker_init_fn, )
        
    
    def _seed_init(self, seed=32767):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        paddle.seed(seed)
    
    def logger(self):
        self.dataset_path = '/mnt/data/Datasets/{}'.format(self.args.dataset)
        time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")
        self.save_dir = os.path.join('{}/{}'.format(self.args.root, self.args.dataset.lower()), f"{self.args.model}_{time_flag}")
    
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.args.best_model_path = os.path.join(self.save_dir, "{}_best.pdparams".format(self.args.model))
        log_path = os.path.join(self.save_dir, "train_{}.log".format(self.args.model))
        self.args.metric_path = os.path.join(self.save_dir, "{}_metrics.csv".format(self.args.model))
        self.args.save_dir = self.save_dir
        print("log save at {}, metric save at {}, weight save at {}".format(log_path, self.args.metric_path, self.args.best_model_path))
        self.args.logger = load_logger(log_path, config=self.args)
        self.log_misc()
        self.args.logger.info("log save at {}, metric save at {}, weight save at {}".format(log_path, self.args.metric_path, self.args.best_model_path))
    
    def log_misc(self):
        if self.args.logger == None:
            return
        self.args.logger.info("Model {}, Datasets {}".format(self.args.model, self.args.dataset))
        self.args.logger.info("lr {}, batch_size {}".format(str(self.args.lr), self.args.batch_size))

    def __call__(self):
        train(self)

            