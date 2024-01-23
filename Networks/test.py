from AURAnet import *
import torch
import numpy as np
import cv2
from util import *
from plot import *
from main import dataloader,normalise_mask_set,dataloader_test
from Metrics import *
from psp_net import  Pspnet
from deeplabv3plus import DeepLab
from u_net import UNet
from resunet import  Resnet34_Unet
#from seg_net import  SegNet
from ATTUNET import  AttUNet
from segnet import SegNet
from DAU_Net import  DAUNet


#from base_ASPP import AURAnet
#from base_SA_CA import AURAnet
#from base_AU import AURAnet


def Predict(X):

    X = X.permute(0, 1, 3, 2)

    pred= model(X)
    #pred = torch.sigmoid(pred)
    avant_norm = pred.data.cpu().numpy()

    return avant_norm

if __name__ =='__main__':
    #model = AURAnet()
    #model = Pspnet(num_classes=1)
    model = DeepLab(num_classes=1)
    #model =  UNet(num_classes=1)
    #model = Resnet34_Unet(3,1)
    #model = SegNet(1)
    #model = AttUNet(in_channel=3, out_channel=1)
    #model  =  DAUNet()
    model.load_state_dict(torch.load("./param/deeplab.pkl"))    # 加载模型参数
    model.eval()
    Miou = ConfusionMatrix(2)

    #to plot all images in the validation set, and you can do the same with the test set using dataloader_test['test'].
    for epoch, (images, masks) in enumerate(dataloader_test['test']):
        truth = masks
        img = images



        proba_prediction =Predict(images)
        #print('###########################################################')
        #print('###########################################################')
        #print('###########################################################')

        tmp = proba_prediction.squeeze()
        normalizedImg = np.zeros((128, 128))  # put the size of your images here
        normalizedImg = cv2.normalize(tmp, normalizedImg, 0, 1, cv2.NORM_MINMAX) #放缩到0和1之间
        mask_pred = normalise_mask_set(normalizedImg, 0.5)



        Miou.update(truth.squeeze(),torch.tensor(mask_pred.squeeze()))
        #xxx = proba_prediction.copy()
        #xxx = 1-xxx



        #plt_save_mask(mask_pred,epoch)
        #plt_save_img(to_numpy(img.permute(0, 3, 2, 1).squeeze()), epoch)



        #plt_save_mask(truth.squeeze(),epoch)
        #plot_side_by_side(to_numpy(truth.squeeze()), mask_pred.squeeze())



        #plot_side_by_side(to_numpy(img.permute(0, 3, 2, 1).squeeze()), proba_prediction.squeeze())
        #plot_heatmaps_alone(xxx.squeeze())



        #print('###########################################################')
        #print('###########################################################')
        #print('###########################################################')
    print(Miou.get_scores())

