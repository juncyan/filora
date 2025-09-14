"""
Lovasz-Softmax and Jaccard hinge loss in Pypaddle
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models.losses import LovaszSoftmaxLoss, BCELoss, CrossEntropyLoss
import numpy as np


def loss_lovasz(cd_map, out1, out2, label1, label2, label_cd):
    
    ce_loss_cd = CrossEntropyLoss()(cd_map, label_cd)
    lovasz_loss_cd = LovaszSoftmaxLoss(ignore_index=255)(cd_map, label_cd)

    ce_loss_clf_t1 = CrossEntropyLoss()(out1, label1)
    lovasz_loss_clf_t1 = LovaszSoftmaxLoss(ignore_index=255)(out1, label1)

    ce_loss_clf_t2 = CrossEntropyLoss()(out2, label2)
    lovasz_loss_clf_t2 = LovaszSoftmaxLoss(ignore_index=255)(out2, label2)

    # Mask for similarity loss (label == 255)
    similarity_mask = (label_cd != 0).unsqueeze(1).expand_as(out1)

    # # Similarity loss calculation (e.g., MSE)
    similarity_loss = F.mse_loss(F.softmax(out1, axis=1) * similarity_mask, F.softmax(out2, axis=1) * similarity_mask, reduction='mean')

    
    final_loss = ce_loss_cd + 0.5 * (ce_loss_clf_t1 + ce_loss_clf_t2 + 0.5 * similarity_loss) + 0.75 * (lovasz_loss_cd + 0.5 * (lovasz_loss_clf_t1 + lovasz_loss_clf_t2))
    # final_loss = ce_loss_cd + ce_loss_clf_t1 + ce_loss_clf_t2  + 0.75 * (lovasz_loss_cd + lovasz_loss_clf_t1 + lovasz_loss_clf_t2)
    return final_loss


def loss(cd_map, out1, out2, label1, label2, label_bn):
    out = paddle.argmax(cd_map, 1).float()
    seg_criterion = CrossEntropyLoss(ignore_index=0)
    loss_seg = seg_criterion(out1, label1.long()) * 0.5 +  seg_criterion(out2, label2.long()) * 0.5
    loss_sc = ChangeSimilarity()(out1[:,1:], out2[:,1:], label_bn)
    loss_bn =  weighted_BCE_logits(out, label_bn)
    final_loss = loss_seg + loss_sc + loss_bn
    return final_loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(paddle.nn.Layer):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

class CrossEntropyLoss2d(nn.Layer):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, axis=1), targets)

def CrossEntropy2d(input, target, weight=None, size_average=False):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    #loss = F.nll_loss(F.log_softmax(input), target, weight=weight, size_average=False)
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss
    
def weighted_BCE(output, target, weight_pos=None, weight_neg=None):
    output = paddle.clamp(output,min=1e-8,max=1-1e-8)
    
    if weight_pos is not None:        
        loss = weight_pos * (target * paddle.log(output)) + \
               weight_neg * ((1 - target) * paddle.log(1 - output))
    else:
        loss = target * paddle.log(output) + (1 - target) * paddle.log(1 - output)

    return paddle.neg(paddle.mean(loss))

def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.25, weight_neg=0.75):
    logit = logit_pixel.reshape([-1])
    truth = truth_pixel.reshape([-1])
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    
    pos = paddle.cast(truth>=0.5, paddle.float32)
    neg = paddle.cast(truth<0.5, paddle.float32)
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos*pos*loss/pos_num + weight_neg*neg*loss/neg_num).sum()

    return loss

class FocalLoss(nn.Layer):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = paddle.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

class FocalLoss2d(nn.Layer):
    def __init__(self, gamma=0, weight=None, size_average=True, ignore_index=-1):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = paddle.create_parameter(self.weight)
        logpt = -F.cross_entropy(input, target, ignore_index=self.ignore_index)
        pt = paddle.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class ChangeSimilarity(nn.Layer):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)
        
    def forward(self, x1, x2, label_change):
        
        b,c,h,w = x1.shape
        x1 = F.softmax(x1, axis=1)
        x2 = F.softmax(x2, axis=1)
        x1 = x1.transpose([0,2,3,1])
        x2 = x2.transpose([0,2,3,1])
        x1 = paddle.reshape(x1,[b*h*w,c])
        x2 = paddle.reshape(x2,[b*h*w,c])
        
        labelmask = paddle.cast(label_change, dtype="bool")
        label_unchange = ~labelmask
        target = paddle.cast(label_unchange, dtype="float32")
        target = target - label_change
        target = paddle.reshape(target,[b*h*w])
        
        loss = self.loss_f(x1, x2, target)
        return loss
        
class ChangeSalience(nn.Layer):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.MSELoss(reduction=reduction)
        
    def forward(self, x1, x2, label_change):
        b,c,h,w = x1.size()
        x1 = F.softmax(x1, dim=1)[:,0,:,:]
        x2 = F.softmax(x2, dim=1)[:,0,:,:]
                
        loss = self.loss_f(x1, x2.detach()) + self.loss_f(x2, x1.detach())
        return loss*0.5
    

def pix_loss(output, target, pix_weight, ignore_index=None):
    # Calculate log probabilities
    if ignore_index is not None:
        active_pos = 1-(target==ignore_index).unsqueeze(1).cuda().float()
        pix_weight *= active_pos
        
    batch_size, _, H, W = output.size()
    logp = F.log_softmax(output, dim=1)
    # Gather log probabilities with respect to target
    target_idx = target.view(batch_size, 1, H, W)
    max_idx = logp.shape[1] - 1
    if paddle.any(target_idx > max_idx):
        target_idx = paddle.clip(target_idx, 0, max_idx)
    logp = logp.gather(1, target_idx)
    # Multiply with weights
    weighted_logp = (logp * pix_weight).view(batch_size, -1)
    # Rescale so that loss is in approx. same interval
    weighted_loss = weighted_logp.sum(1) / pix_weight.view(batch_size, -1).sum(1)
    # Average over mini-batch
    weighted_loss = -1.0 * weighted_loss.mean()
    return weighted_loss

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = paddle.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Layer):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = paddle.sum(paddle.mul(predict, target), dim=1) + self.smooth
        den = paddle.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Layer):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]