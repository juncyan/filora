import logging
import os
import sys
import numpy as np
import pandas as pd
import re
import glob
from logging.handlers import RotatingFileHandler

__all__ = ['setup_logger', 'load_logger']


# reference from: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/logger.py
def setup_logger(name, save_dir, distributed_rank, filename="log.txt", mode='w'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def load_logger(save_log_dir,save=True,print=True,config=None):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                fmt='[ %(asctime)s ] %(message)s')
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(save_log_dir, maxBytes=1024000, backupCount=5)
    formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                fmt='[ %(asctime)s ] %(message)s')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if config != None:
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger

def find_max_iter(logcontents, rows):
    max_iter = 1
    total_iter = 1
    find_max_iter = False
    find_total_iter = False

    for i in range(rows):
        idx = int(-i)
        if (find_max_iter == False) and ("[TRAIN]" in logcontents[idx]) and ("best iter" in logcontents[idx]):
            ts = logcontents[idx].split("best iter")[-1]
            max_iter = int(re.findall(r"\d+", ts)[0])
            find_max_iter = True

        if (find_total_iter == False) and ("[TRAIN]" in logcontents[idx]) and ("iter:" in logcontents[idx]):
            ts = logcontents[idx].split("iter:")[-1]
            res = re.findall(r"\d+\/\d+", ts)[0]
            total_iter = int(res.split("/")[-1])
            find_total_iter = True

        # print(find_max_iter, find_total_iter)
        # print(max_iter, total_iter)
        if find_max_iter and find_total_iter:
            return max_iter, total_iter

def extract_data(file_path):
    with open(file_path, 'r') as logfile:
        logcontents = logfile.readlines()
    rows = len(logcontents)

    max_iter, total_iter = find_max_iter(logcontents, rows)
    find_max_local = 1
    for i in range(rows):
        idx = i if total_iter / max_iter > 2 else int(-i)
        partstr = f"{max_iter}/{total_iter}"
        if partstr in logcontents[idx]:
            find_max_local = idx if idx > 0 else rows + idx
            break

    dret = {}
    for i in range(find_max_local + 2, find_max_local + 6):
        ts = logcontents[i]
        if "[EVAL]" in ts:
            res = ts.split("[EVAL]")[-1]
            # res = res.replace(" ", "")
            res = res.replace("\n", "")
            items = res.split(",")
            d = dict(item.split(":") for item in items)
            dret.update(d)
    return dret

def extract_data_as_array(file_path):
    print(file_path)
    ret = extract_data(file_path)
    iou = re.findall("0\.\d+", ret[" Class IoU"])

    precision = re.findall("0\.\d+", ret[" Class Precision"])
    recall = re.findall("0\.\d+", ret[" Class Recall"])

    acc = ret[" Acc"]
    miou = ret["mIoU"]
    kappa = ret["kappa"]
    f1 = ret["Macro_f1"]

    array = np.array([float(iou[0]), float(iou[1]), float(miou), float(precision[0]), float(precision[1]),
                      float(acc), float(recall[0]), float(recall[1]), float(kappa), float(f1)])
    return array

def save_log_as_csv(data_dir, save_path):
    keys = ["iou1", "iou2", "miou", "acc1", "acc2", "macc", "recall1", "recall2", "kappa", "F1"]
    res = {}
    folders = os.listdir(data_dir)
    idx = 0
    for f in folders:
        fpath = os.path.join(data_dir, f)
        mn = f.split("_")[0]
        array = None
        try:
            if os.path.isdir(fpath):
                txt_files = glob.glob(os.path.join(fpath, '*.log'))[0]
                # print(txt_files)
                array = extract_data_as_array(txt_files)
            res.update({f"{mn}{idx}": array})
            idx += 1
        except:
            pass

    d = pd.DataFrame(res)
    indexs = d.keys()
    print(keys)
    data = d.values.transpose([1, 0])
    s = pd.DataFrame(data, indexs, keys)
    s.to_csv(save_path)