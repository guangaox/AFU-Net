import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import numpy as np
import copy
import time

from AFU_Net import *

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.autograd import Variable as V
import torch.optim as optim
from torch.optim import lr_scheduler
import glob
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
import PIL.Image as Image
from PIL import Image
from libtiff import TIFF
from Metrics import *

from util import tif_to_nparray,create_dir,to_numpy,resize_my_images,load_image,load_set,save_to_tif
from albumentations import (
    Resize,
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma
)


path_to_img = r'C:\Users\xga\Desktop\SAR\IMG_train'
path_to_mask = r'C:\Users\xga\Desktop\SAR\LAB_train'

path_to_test_img = r"C:\Users\xga\Desktop\SAR\LAB_train"
path_to_test_mask = r"C:\Users\xga\Desktop\SAR\LAB_train"



transform_train=Compose([
    VerticalFlip(p=0.5), #沿X轴进行翻转
    RandomRotate90(p=0.5), # 随机旋转90度
    Resize(256,256),
    OneOf([
        ElasticTransform(p=0.8, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(p=0.2),
        OpticalDistortion(p=0.4, distort_limit=2, shift_limit=0.5)
        ], p=0.8)
         ])

transform_val=Compose([
   Resize(256,256)])#depends of your image sizes
transform_test=Compose([
   Resize(256,256)])#depends of your image sizes
transform_nolastic=Compose([
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5)])



class Embryo_elastic(Dataset):
    def __init__(self, img_fol, mask_fol, transform=None):
        self.img_fol = img_fol
        self.mask_fol = mask_fol
        self.transform = transform

    def __getitem__(self, idx):
        image = load_set(self.img_fol, is_mask=False)[0][idx]
        mask = load_set(self.mask_fol, is_mask=True)[0][idx]
        if self.transform:

            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            #print(image.shape)
            mask = augmented['mask']
            # change the normalize img by the normalization you want to do
            normalize_img = transforms.Compose([transforms.ToTensor()])
            image = normalize_img(image)
            image = image.permute(0, 2, 1)

            transform_to_tensor = transforms.Compose([transforms.ToTensor()])
            mask = transform_to_tensor(mask)


        else:
            # same here

            normalize_img = transforms.Compose([transforms.ToTensor()])
            image = normalize_img(image)
            image = image.permute(0, 2, 1)
            transform_to_tensor = transforms.Compose([transforms.ToTensor()])
            mask = transform_to_tensor(mask)

        return image, mask

    def __len__(self):

        return len(load_set(self.mask_fol, is_mask=True)[1])

def get_emb_elastic_loader(path_img, path_mask, validation_split=0, shuffle_dataset=True):
    dataset = Embryo_elastic(path_img, path_mask)  # instantiating the data set.

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split_val = int(np.floor(validation_split * dataset_size)) # 0.2*34 = 6.8 = 6
    if shuffle_dataset:
        np.random.shuffle(indices)

    train_indices = indices[split_val:] #0.8用于训练
    val_indices = indices[: split_val]  # 0.2用于验证

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    dataset_train = Embryo_elastic(path_img, path_mask, transform=transform_train)

    dataset_val = Embryo_elastic(path_img, path_mask, transform=transform_val)

    loader = {
        'train': DataLoader(dataset_train, batch_size=4, sampler=train_sampler,drop_last=True),
        'val': DataLoader(dataset_val, batch_size=1, sampler=valid_sampler,drop_last=True),
    }
    return loader

dataloader=get_emb_elastic_loader(path_to_img,path_to_mask)


# Dataloader for testing
class Embryo_elastic_for_test(Dataset):
    def __init__(self, img_fol, mask_fol):
        self.img_fol = img_fol
        self.mask_fol = mask_fol

    def __getitem__(self, idx):
        image = load_set(self.img_fol, is_mask=False)[0][idx]
        mask = load_set(self.mask_fol, is_mask=True)[0][idx]
        # add normalization if you want
        augmented = transform_test(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        normalize_img = transforms.Compose([transforms.ToTensor()])
        image = normalize_img(image)
        image = image.permute(0, 2, 1)

        transform_to_tensor = transforms.Compose([transforms.ToTensor()])
        mask = transform_to_tensor(mask)

        return image, mask

    def __len__(self):
        return len(load_set(self.img_fol, is_mask=False)[1])


def get_dataset_test(path_img_test, path_mask_test):
    dataset_test = Embryo_elastic_for_test(path_img_test, path_mask_test)
    loader = {
        'test': DataLoader(dataset_test, batch_size=1)
    }
    return loader

dataloader_test=get_dataset_test(path_to_test_img,path_to_test_mask)

''' training '''

from collections import defaultdict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_loss(pred, target, metrics, bce_weight=0.5):  # you can use the weights you want for the bce_weights
    #bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
    bce = floss(pred,target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    AC = active_contour_loss(target, pred)

    loss_bce_dice = bce * bce_weight + dice * (1 - bce_weight)

    #loss = 0.75 * loss_bce_dice + 0.25 * AC  # put whatever weights you want
    #loss = 0.5 * loss_bce_dice + 0.5 * AC
    loss_bce_ac = 0.5*AC+0.5*bce
    loss = loss_bce_dice
    #loss = loss_bce_ac
    #loss = bce
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['AC'] += AC.data.cpu().numpy()  * target.size(0)
    metrics['loss_bce_dice'] += loss_bce_dice.data.cpu().numpy() * target.size(0)
    #metrics['loss_bce_ac'] += loss_bce_ac.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

floss = FocalLoss()

def active_contour_loss(y_true, y_pred):
    '''
    y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
    weight: scalar, length term weight.

    '''
    # length term
    delta_r = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
    delta_c = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)

    delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
    delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
    delta_pred = torch.abs(delta_r + delta_c)

    epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
    lenth = torch.mean(torch.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.

    # region term
    C_in = torch.ones_like(y_pred)
    C_out = torch.zeros_like(y_pred)
# eq3
    # C_in 为1时，说明在分割目标内部，y_pred预测内部时损失已经为0，[预测外部时 region_in此时等于torch.mean(y_pred) 要让y_pred尽量为0]
    # C_out 为0时，说明在分割目标外部，y_pred预测外部时损失已经为0，预测内部时 region_out此时等于torch.mean(y_pred) 要让y_pred尽量为1
    # 本质就是在内外部分别进行交叉熵损失
    region_in = torch.mean(y_pred * (y_true - C_in) ** 2)  # equ.(12) in the paper, mean is used instead of sum.
    region_out = torch.mean((1 - y_pred) * (y_true - C_out) ** 2)

    region = region_in + region_out

    loss = 0.2 * lenth + 0.8 * region

    return loss


def compute_metrics(metrics, epoch_samples):
    computed_metrics = {}
    for k in metrics.keys():
        computed_metrics[k] = metrics[k] / epoch_samples
    return computed_metrics


def print_metrics(computed_metrics, phase):
    outputs = []
    for k in computed_metrics.keys():
        outputs.append("{}:{:4f}".format(k, computed_metrics[k]))

    print("\t{}-> {}".format(phase.ljust(5), "|".join(outputs)))


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) /
                 (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def normalise_mask_set(mask, threshold):
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    return mask


def normalise_mask(mask, threshold=0.5):
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    return mask


def metrics_line(data):
    phases = list(data.keys())
    metrics = list(data[phases[0]][0].keys())

    i = 0
    fig, axs = plt.subplots(1, len(metrics))
    fig.set_figheight(6)
    fig.set_figwidth(6 * len(metrics))
    for metric in metrics: #bce dice ac
        for phase in phases: # train or val

            axs[i].plot([i[metric] for i in data[phase]], label=phase)

        axs[i].set_title(metric)

        i += 1
    plt.legend()
    plt.show()



class Trainers(object):

    def __init__(self, model, optimizer=None, scheduler=None):

        super().__init__()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)

        self.optimizer = optimizer
        if self.optimizer == None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.scheduler = scheduler
        if self.scheduler == None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5, gamma=0.1)

    def train_model(self, dataloaders, num_epochs=25):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10
        epochs_metrics = {
            'train': [],
            #'val': []
        }
        #Miou = ConfusionMatrix(2)
        for epoch in range(num_epochs):
            print('Epoch {}/{}:'.format(epoch + 1, num_epochs))
            #Miou = mIOU(2)

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train']:
                if phase == 'train':
                    for param_group in self.optimizer.param_groups:
                        print("\tlearning rate: {:.2e}".format(
                            param_group['lr']))

                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                metrics = defaultdict(float)
                epoch_samples = 0
                for inputs, labels in dataloaders[phase]:
                    #B C H W

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device) # B 1 H W  C=1
                    inputs = inputs.permute(0, 1, 3, 2)  # put the mask and inputs in the same settings

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        outputs = self.model(inputs) # B 1 H W
                        loss = calc_loss(outputs, labels, metrics)
                        # backward + optimize only if in training phase



                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()


                    # statistics
                    epoch_samples += inputs.size(0)


                computed_metrics = compute_metrics(metrics, epoch_samples)
                print_metrics(computed_metrics, phase)
                epochs_metrics[phase].append(computed_metrics)
                epoch_loss = metrics['loss'] / epoch_samples

                if phase == 'train':
                    self.scheduler.step()

                # deep copy the model

                if phase == 'train' and epoch_loss < best_loss:
                    print("\tSaving best model, epoch loss {:4f} < best loss {:4f}".format(
                        epoch_loss, best_loss))
                    best_loss = epoch_loss
                    #best_model_wts = copy.deepcopy(self.model.state_dict())
                    #torch.save(self.model.state_dict(), "./param/segnet.pkl")
                    #torch.save(self.model, PATH)


            time_elapsed = time.time() - since
            print('\t{:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('-' * 10)
        best_model_wts = copy.deepcopy(self.model.state_dict())
        torch.save(self.model.state_dict(), "./param/u5net.pkl")

            #Miou.reset()
        print('Best train loss: {:4f}'.format(best_loss))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        #       画     出        损      失

        metrics_line(epochs_metrics)

    def predict(self, X):

        self.model.eval()
        X = X.permute(0, 1, 3, 2)
        inputs = X.to(self.device)
        pred = self.model(inputs)
        avant_norm = pred.data.cpu().numpy()

        return avant_norm
if __name__ == '__main__':

    learning_rate=1e-4
    step_size=40
    gamma=0.1
    num_epochs=30
    model = AFU_Net()

    model.train()

    optimizer_func = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer_func, step_size=step_size, gamma=gamma)
    trainer = Trainers(model, optimizer=optimizer_func, scheduler=scheduler)
    trainer.train_model(dataloader, num_epochs=num_epochs)
#to plot all images in the validation set, and you can do the same with the test set using dataloader_test['test'].
    Miou = ConfusionMatrix(2)
    for epoch, (images, masks) in enumerate(dataloader_test['test']):
        truth = masks
        img = images
        proba_prediction = trainer.predict(images)
        #print('###########################################################')
        #print('###########################################################')
        #print('###########################################################')
        tmp = proba_prediction.squeeze()
        normalizedImg = np.zeros((128, 128))  # put the size of your images here
        normalizedImg = cv2.normalize(tmp, normalizedImg, 0, 1, cv2.NORM_MINMAX)
        mask_pred = normalise_mask_set(normalizedImg, 0.5)


        Miou.update(truth.squeeze(),torch.tensor(mask_pred.squeeze()))


        #plot_side_by_side(to_numpy(truth.squeeze()), mask_pred.squeeze())
        #plot_side_by_side(to_numpy(img.permute(0, 3, 2, 1).squeeze()), proba_prediction.squeeze())

        #print('###########################################################')
        #print('###########################################################')
        #print('###########################################################')
    print(Miou.get_scores())
