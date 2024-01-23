import torch
import numpy as np

np.seterr(divide="ignore", invalid="ignore")

class mIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    # 返回的是混淆矩阵
    def _fast_hist(self, label_pred, label_true):
        # 去除背景
        # ground truth中所有正确(值在[0, classe_num])的像素label的mask
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # 计算出每一类（0-n**2-1）中对应的数（0-n**2-1）出现的次数，返回值为（n，n）
        # confusion_matrix是一个[num_classes, num_classes]的矩阵,
        # confusion_matrix矩阵中(x, y)位置的元素代表该张图片中真实类别为x, 被预测为y的像素个数
        '''
        关于下面的混淆矩阵如何计算出来的可能会有些初学者不大理解，笔者根据自己的想法对下面的代码有一定的见解，
        可能有一定错误，欢迎指出
        我们之前得到的是两张由0-num_class-1的数字组成的label，分别对应我们的类别总数
        self.num_classes * label_true[mask].astype(int)，这段代码通过将label_true[mask]乘以num_class
        第0类还是0，第一类的数字变成num_class（注意这是在原来的图上操作）,以此类推，
        +label_pred[mask]，对于这一步我举个栗子，比如groundtrue是第一类，num_class=21,在之前操作已经将该像素块
        变成21了，如果我预测的还是第一类，则这一像素块变成了22，在bincount函数中，使得数字22的次数增加了1，在后面的reshape中
        数字22对于的就是第二行第第二列，也就是对角线上的（因为混淆矩阵的定义就是对角线上的就是预测正确的，即TP），所以得到了
        hist就是混淆矩阵
        '''

        hist = np.bincount(
            self.num_classes * label_true[mask].type(torch.int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        predictions = predictions.cpu()
        gts = gts.cpu()
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
    def evaluate(self):
        '''
        miou = TP / (TP+FP+FN)
        因此下面的式子显然是计算miou的
        :return:
        '''
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return np.nanmean(iu[1:])
    def reset(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))
        # self.iou = []
        # self.iou_threshold = []

# TP = np.diag(self.hist)
#

class ConfusionMatrix(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        # axis = 0: target
        # axis = 1: prediction
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        # self.iou = []
        # self.iou_threshold = []

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)

        hist = np.bincount(n_class * label_true[mask].type(torch.int) + label_pred[mask].type(torch.int),
                           minlength=n_class ** 2).reshape(2,2)

        return hist

    def update(self, label_trues, label_preds):
        #label_preds =label_preds>0.5
        for lt, lp in zip(label_trues, label_preds):
            tmp = self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
            # iu = np.diag(tmp) / (tmp.sum(axis=1) + tmp.sum(axis=0) - np.diag(tmp))
            # self.iou.append(iu[1])
            # if iu[1] >= 0.65: self.iou_threshold.append(iu[1])
            # else: self.iou_threshold.append(0)

            self.confusion_matrix += tmp

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc

        """
        # self.hist.sum(axis=0) = [TP+FP,FN+TN]
        # self.hist.sum(axis=1) = [TP+FN,FP+TN]
        # np.diag(self.hist) = [TP,TN]
        # miou = TP/(TP+FN+FP)
        # Accuacy(准确率) = (TP+TN)/(TP+TN+FP+FN)
        # Precision(精准率) = TP/(TP+FP)
        # Recall(召回率) = TP/(TP+FN)
        hist = self.confusion_matrix
        # accuracy is recall/sensitivity for each class, predicted TP / all real positives
        # axis in sum: perform summation along
        acc = np.nan_to_num(np.diag(hist) / hist.sum(axis=1))
        acc_mean = np.mean(np.nan_to_num(acc))

        intersect = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        iou = intersect / union
        mean_iou = np.mean(np.nan_to_num(iou))

        freq = hist.sum(axis=1) / hist.sum()  # freq of each target   # (TP+TN)/(TP
        # fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        freq_iou = (freq * iou).sum()

        Precision = np.nan_to_num(np.diag(hist)/hist.sum(axis=0))
        Recall = np.nan_to_num(np.diag(hist)/hist.sum(axis=1))


        return {'0A': acc,
                'OA_mean': acc_mean,
                'freqw_iou': freq_iou,
                'iou': np.nan_to_num(iou),
                'iou_mean': np.nan_to_num(mean_iou),
                'Precision': Precision,
                'Recall': Recall,
                'F1':np.nan_to_num((2*Precision*Recall)/(Precision+Recall))
                # 'IoU_threshold': np.mean(np.nan_to_num(self.iou_threshold)),
                }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        # self.iou = []
        # self.iou_threshold = []


import numpy as np


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        # 找出标签中需要计算的类别,去掉了背景
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        hist = np.bincount(
            self.num_classes * label_true[mask].type(torch.int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    # 输入：预测值和真实值
    # 语义分割的任务是为每个像素点分配一个label
    def evaluate(self, predictions, gts):

        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        # self.hist.sum(axis=0) = [TP+FP,FN+TN]
        # self.hist.sum(axis=1) = [TP+FN,FP+TN]
        # np.diag(self.hist) = [TP,TN]
        # miou = TP/(TP+FN+FP)
        # Accuacy(准确率) = (TP+TN)/(TP+TN+FP+FN)
        # Precision(精准率) = TP/(TP+FP)
        # Recall(召回率) = TP/(TP+FN)
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        Precision = np.diag(self.hist)/self.hist.sum(axis=0)
        Recall = np.diag(self.hist)/self.hist.sum(axis=1)

        # -----------------其他指标------------------------------
        # mean acc

        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))

        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

        return acc, acc_cls, iou, miou, fwavacc
















if __name__ == '__main__':
    IOU = ConfusionMatrix(2)
    x1 = PIL.Image
    IOU.update(x1,y1)
    print(IOU.get_scores())








