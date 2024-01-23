import torch
import numpy as np

np.seterr(divide="ignore", invalid="ignore")



class ConfusionMatrix(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes

        self.confusion_matrix = np.zeros((n_classes, n_classes))


    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)

        hist = np.bincount(n_class * label_true[mask].type(torch.int) + label_pred[mask].type(torch.int),
                           minlength=n_class ** 2).reshape(2,2)

        return hist

    def update(self, label_trues, label_preds):

        for lt, lp in zip(label_trues, label_preds):
            tmp = self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

            self.confusion_matrix += tmp

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc

        """

        hist = self.confusion_matrix

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


    def evaluate(self, predictions, gts):

        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        Precision = np.diag(self.hist)/self.hist.sum(axis=0)
        Recall = np.diag(self.hist)/self.hist.sum(axis=1)



        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))

        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

        return acc, acc_cls, iou, miou, fwavacc

















