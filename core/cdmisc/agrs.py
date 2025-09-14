import datetime
import os
import argparse
from .logger import load_logger

__all__ = ['Args', 'parse_args']

class Args():
    def __init__(self, dst_dir, model_name):
        # [epoch, loss, acc, miou, mdice,kappa,macro_f1]
        self.img_ab_concat = False
        self.en_load_edge = False
        self.num_classes = 0
        self.batch_size = 0
        self.iters = 0
        self.device = "gpu:0"

        self.pred_idx = 0
        self.data_name = ""
        time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")
        self.save_dir = os.path.join(dst_dir, f"{model_name}_{time_flag}")
        self.model_name = model_name

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # self.save_predict = os.path.join(self.save_dir , "predict")
        # if not os.path.exists(self.save_predict):
        #     os.makedirs(self.save_predict)

        self.best_model_path = os.path.join(self.save_dir, "{}_best.pdparams".format(model_name))
        log_path = os.path.join(self.save_dir, "train_{}.log".format(model_name))
        self.metric_path = os.path.join(self.save_dir, "{}_metrics.csv".format(model_name))
        print("log save at {}, metric save at {}, weight save at {}".format(log_path, self.metric_path, self.best_model_path))
        self.epoch = 0
        self.loss = 0
        self.logger = load_logger(log_path)
        # self.logger = LogWriter(logdir=self.save_dir)
        # self.logger.add_text("starts","log save at {}, metric save at {}, weight save at {}".format(log_path, self.metric_path, self.best_model_path))
        # self.logger.add_scalars()
        self.logger.info("log save at {}, metric save at {}, weight save at {}".format(log_path, self.metric_path, self.best_model_path))
        # writer_csv(self.metric_path, headers=demo_predict_data_headers)


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Overfitting Test')
    # model
    parser.add_argument('--model', type=str, default='msfgnet',
                        help='model name (default: msfgnet)')
    parser.add_argument('--device', type=str, default='gpu:0',
                        choices=['gpu:0', 'gpu:1', 'cpu'],
                        help='device (default: gpu:0)')
    parser.add_argument('--dataset', type=str, default='LEVIR_CD',
                        help='dataset name (default: LEVIR_CD)')
    parser.add_argument('--iters', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--img_ab_concat', type=bool, default=False,
                        help='img_ab_concat False')
    parser.add_argument('--en_load_edge', type=bool, default=False,
                        help='en_load_edge False')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='num classes (default: 2)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size (default: 4)')
    
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    
    parser.add_argument('--num_works', type=int, default=8,
                        help='num_works (default: 8)')
    args = parser.parse_args()
    return args