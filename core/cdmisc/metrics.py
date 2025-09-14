import copyreg
import types
import numpy as np

class Metrics(object):
    def __init__(self, num_class):
        self.__num_class = num_class
        self.__confusion_matrix = np.zeros((self.__num_class,) * 2)

        self.__TP = 0.#np.diag(self.__confusion_matrix)
        self.__RealN = 0.#np.sum(self.__confusion_matrix, axis=0)  # TP+FN
        self.__RealP = 0.#np.sum(self.__confusion_matrix, axis=1)  # TP+FP
        self.__sum = 0.#np.sum(self.__confusion_matrix)

    def Pixel_Accuracy(self):
        Acc = self.__TP.sum() / self.__sum
        return Acc

    def Class_Precision(self):
        #TP/TP+FP
        precision = self.__TP / (self.__RealP + 1e-5)
        # Acc = np.nanmean(Acc)
        return precision

    def Intersection_over_Union(self):
        IoU = self.__TP / (1e-5 + self.__RealP + self.__RealN - self.__TP)
        return IoU

    def Mean_Intersection_over_Union(self):
        IoU = self.Intersection_over_Union()
        MIoU = np.nanmean(IoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = self.__RealP / self.__sum
        iu = self.__TP / (1e-5 + self.__RealP + self.__RealN - self.__TP)
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Kappa(self):
        P0 = self.Pixel_Accuracy()
        Pe = np.sum(self.__RealP * self.__RealN) / (self.__sum * self.__sum)
        return (P0 - Pe) / (1 - Pe)

    def F1_score(self, belta=1):
        precision = self.Class_Precision()
        recall = self.Recall()
        f1_score = (1 + belta * belta) * precision * recall / (belta * belta * precision + recall + 1e-5)
        return f1_score

    def Macro_F1(self, belta=1):
        return np.nanmean(self.F1_score(belta))

    def Dice(self):
        dice = 2 * self.__TP / (self.__RealN + self.__RealP + 1e-5)
        return dice

    def Mean_Dice(self):
        dice = self.Dice()
        return np.nanmean(dice)

    def Recall(self):  # 预测为正确的像素中确认为正确像素的个数
        #TP/ TP+FN
        recall = self.__TP / (self.__RealN + 1e-5)
        return recall
    
    def Get_Metric(self):
        self.calc()
        pa = np.round(self.Pixel_Accuracy(),4)
        iou = np.round(self.Intersection_over_Union(),4)
        miou = np.round(np.nanmean(iou),4)
        prices = np.round(self.Class_Precision(),4)
        f1 = np.round(self.F1_score(),4)
        mf1 = np.round(np.nanmean(f1),4)
        recall = np.round(self.Recall(),4)
        Pe = np.round(np.sum(self.__RealP * self.__RealN) / (self.__sum * self.__sum),4)
        kappa =  np.round((pa - Pe) / (1 - Pe),4)

        cls_iou = dict(zip(['iou_'+str(i) for i in range(self.__num_class)], iou))
        cls_precision = dict(zip(['precision_'+str(i) for i in range(self.__num_class)], prices))
        cls_recall = dict(zip(['recall_'+str(i) for i in range(self.__num_class)], recall))
        cls_F1 = dict(zip(['F1_'+str(i) for i in range(self.__num_class)], f1))

        metrics ={"pa":pa, "miou": miou, "mf1":mf1, "kappa":kappa}
        metrics.update(cls_iou)
        metrics.update(cls_F1)
        metrics.update(cls_precision)
        metrics.update(cls_recall)
        self.reset()
        return metrics

    def __generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.__num_class)
        label = self.__num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.__num_class ** 2)
        confusion_matrix = count.reshape(self.__num_class, self.__num_class)
        return confusion_matrix

    # def add_batch(self, gt_image, pre_image):
    def add_batch(self, pred, lab):
        pred = np.array(pred)
        lab = np.array(lab)
        # print(pred.shape, lab.shape)
        if len(lab.shape) == 4 and lab.shape[1] != 1:
            lab = np.argmax(lab, axis=1)

        if len(pred.shape) == 4 and pred.shape[1] != 1:
            pred = np.argmax(pred, axis=1)

        gt_image = np.squeeze(lab)
        pre_image = np.squeeze(pred)
        
        assert (np.max(pre_image) <= self.__num_class)
        # assert (len(gt_image) == len(pre_image))
        # print(gt_image.shape)
        # print(pre_image.shape)
        assert gt_image.shape == pre_image.shape
        self.__confusion_matrix += self.__generate_matrix(gt_image, pre_image)
        
    def calc(self):
        self.__TP = np.diag(self.__confusion_matrix)
        self.__RealN = np.sum(self.__confusion_matrix, axis=0)  # TP+FN
        self.__RealP = np.sum(self.__confusion_matrix, axis=1)  # TP+FP
        self.__sum = np.sum(self.__confusion_matrix)

    def reset(self):
        self.__confusion_matrix = np.zeros((self.__num_class,) * 2)
        self.__TP = 0.     #TP
        self.__RealN = 0.  # TP+FN
        self.__RealP = 0.  # TP+FP
        self.__sum = 0.  # np.sum(self.__confusion_matrix)


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_


def Acc_Metric(Mask, GT):
    GT_pos_sum = np.sum(GT == 1)
    Mask_pos_sum = np.sum(Mask == 1)
    True_pos_sum = np.sum((GT == 1) * (Mask == 1))
    Precision = float(True_pos_sum) / (Mask_pos_sum + 1e-6)
    Recall = float(True_pos_sum) / (GT_pos_sum + 1e-6)
    IoU = float(True_pos_sum) / (GT_pos_sum + Mask_pos_sum - True_pos_sum + 1e-6)
    # IoU = Precision * Recall / (Precision + Recall - Precision * Recall + 1e-6)
    if GT_pos_sum == 0 and Mask_pos_sum == 0:
        IoU = 1
    F1_score = 2 * Precision * Recall / (Precision + Recall + 1e-6)
    # return Recall,Precision,F1_score,IoU
    return F1_score


def Pixel_A(Mask, GT):
    tp = np.sum(np.logical_and(Mask == 1, GT == 1))
    fp = np.sum(np.logical_and(Mask == 1, GT != 1))
    # tn = np.sum(np.logical_and(Mask != 1, GT != 1))
    fn = np.sum(np.logical_and(Mask != 1, GT == 1))
    return tp, fp, fn


# def Acc_Metric(Mask,GT):
#     GT_pos_sum = np.sum(GT == 1)
#     Mask_pos_sum = np.sum(Mask == 1)
#     True_pos_sum = np.sum((GT == 1) * (Mask == 1))
#     Precision = float(True_pos_sum) / (Mask_pos_sum + 1e-6)
#     Recall = float(True_pos_sum) / (GT_pos_sum + 1e-6)
#     IoU = float(True_pos_sum) / (GT_pos_sum + Mask_pos_sum - True_pos_sum + 1e-6)
#     # IoU = Precision * Recall / (Precision + Recall - Precision * Recall + 1e-6)
#     if GT_pos_sum==0 and Mask_pos_sum==0:
#         IoU =1
#     F1_score = 2 * Precision * Recall / (Precision + Recall + 1e-6)
#     return Recall,Precision,F1_score,IoU


def Modify_Lable(Mask, GT):
    error = np.abs(GT - Mask)
    error[error > 0.85] = 1
    error[error <= 0.85] = 0
    return np.abs(GT - error)


def r_iou(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(predict == 1)
    fn = np.sum(label == 1)
    if (fp + fn - tp) == 0:
        return 1
    return tp / (fp + fn - tp)


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_


def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        print("DiffDim: Different dimensions of matrices!")


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


copyreg.pickle(types.MethodType, _pickle_method)


class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None, ignore_label=255):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))
        self.ignore_label = ignore_label

    def add(self, gt, pred):
        assert (np.max(pred) <= self.nclass)
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == self.ignore_label:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert (matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    # Pii为预测正确的数量，Pij和Pji分别被解释为假正和假负，尽管两者都是假正与假负之和
    def recall(self):  # 预测为正确的像素中确认为正确像素的个数
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall / self.nclass

    def accuracy(self):  # 分割正确的像素除以总像素
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy / self.nclass

    # 雅卡尔指数，又称为交并比（IOU）
    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            if not self.M[i, i] == 0:
                jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass) / len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass:  # and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m


def get_iou(data_list, class_num, save_path=None):
    """
    Args:
      data_list: a list, its elements [gt, output]
      class_num: the number of label
    """
    from multiprocessing import Pool

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    # print(j_list)
    # print(M)
    # print('meanIOU: ' + str(aveJ) + '\n')

    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list) + '\n')
            f.write(str(M) + '\n')
    return aveJ, j_list


class Evaluator(object):
    def __init__(self, num_class):
        self.__num_class = num_class
        self.__confusion_matrix = np.zeros((self.__num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.__confusion_matrix).sum() / self.__confusion_matrix.sum()
        return Acc

    def Class_Precision(self):
        #TP/TP+FP
        precision = np.diagonal(self.__confusion_matrix) / (self.__confusion_matrix.sum(axis=1) + 1e-5)
        # Acc = np.nanmean(Acc)
        return precision

    def Intersection_over_Union(self):
        IoU = np.diagonal(self.__confusion_matrix) / (1e-5 +
                                                np.sum(self.__confusion_matrix, axis=1) + np.sum(self.__confusion_matrix,axis=0) -
                                                np.diagonal(self.__confusion_matrix))
        return IoU

    def Mean_Intersection_over_Union(self):
        # MIoU = np.diag(self.__confusion_matrix) / (
        #             np.sum(self.__confusion_matrix, axis=1) + np.sum(self.__confusion_matrix, axis=0) -
        #             np.diag(self.__confusion_matrix))
        # MIoU = np.nanmean(MIoU)
        IoU = self.Intersection_over_Union()
        MIoU = np.nanmean(IoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.__confusion_matrix, axis=1) / np.sum(self.__confusion_matrix)
        iu = np.diag(self.__confusion_matrix) / (1e-5 +
                                               np.sum(self.__confusion_matrix, axis=1) + np.sum(self.__confusion_matrix,
                                                                                              axis=0) -
                                               np.diag(self.__confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Kappa(self):
        diag = np.diag(self.__confusion_matrix)
        clo_sum = np.sum(self.__confusion_matrix, axis=0)
        row_sum = np.sum(self.__confusion_matrix, axis=1)
        num = np.sum(self.__confusion_matrix)
        P0 = np.sum(diag) / num
        Pe = np.sum(row_sum * clo_sum) / (num * num)
        return (P0 - Pe) / (1 - Pe + 1e-5)

    def F1_score(self, belta=1):
        TP = np.diag(self.__confusion_matrix)
        RealN = np.sum(self.__confusion_matrix, axis=0)  # TP+FN
        RealP = np.sum(self.__confusion_matrix, axis=1)  # TP+FP
        precision = TP / (RealP + 1e-5)
        recall = TP / (RealN + 1e-5)
        f1_score = (1 + belta * belta) * precision * recall / (belta * belta * precision + recall + 1e-5)
        return f1_score

    def Macro_F1(self, belta=1):
        return np.nanmean(self.F1_score(belta))

    def Dice(self):
        dice = 2 * np.diag(self.__confusion_matrix) / (np.sum(self.__confusion_matrix, axis=0) + np.sum(self.__confusion_matrix, axis=1))
        return dice

    def Mean_Dice(self):
        dice = self.Dice()
        return np.nanmean(dice)

    def Recall(self):  # 预测为正确的像素中确认为正确像素的个数
        #TP/ TP+FN
        recall = np.diagonal(self.__confusion_matrix) / (np.sum(self.__confusion_matrix, axis=0) + 1e-5)
        return recall

    def __generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.__num_class)
        label = self.__num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.__num_class ** 2)
        __confusion_matrix = count.reshape(self.__num_class, self.__num_class)
        # print("__confusion_matrix:\n", __confusion_matrix)
        return __confusion_matrix

    # def add_batch(self, gt_image, pre_image):
    def add_batch(self, pred, lab):
        pred = np.array(pred.cpu())
        lab = np.array(lab.cpu())

        if len(lab.shape) == 4:
            lab = np.argmax(lab, axis=1)

        if len(pred.shape) == 4:
            pred = np.argmax(pred, axis=1)

        gt_image = np.squeeze(lab)
        pre_image = np.squeeze(pred)

        assert (np.max(pre_image) <= self.__num_class)
        assert (len(gt_image) == len(pre_image))
        # print(gt_image.shape)
        # print(pre_image.shape)
        assert gt_image.shape == pre_image.shape
        self.__confusion_matrix += self.__generate_matrix(gt_image, pre_image)

    def reset(self):
        self.__confusion_matrix = np.zeros((self.__num_class,) * 2)


if __name__ == "__main__":
    print("TNet.Metrics run")
    x = [0.83611815, 0.51126306, 0.69839933, 0.75191475]
    print(np.mean(x))





