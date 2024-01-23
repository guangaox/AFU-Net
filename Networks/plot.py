# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:44:43 2020

@author: ethan
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os 
from libtiff import TIFF
import cv2 as cv
path = r"C:\Users\xga\Desktop\ftest\testimg\\"
Path = r"C:\Users\xga\Desktop\ftest\testimg\\"
def plot_side_by_side(im1,im2):
  f = plt.figure()
  plt.set_cmap('binary')
  f.add_subplot(1,2, 1)
  plt.imshow(np.rot90(im1,2))
  f.add_subplot(1,2, 2)
  plt.imshow(np.rot90(im2,2))
  plt.show(block=True)


def plot_heatmaps_alone(img):
    f = plt.figure()
    plt.imshow(img)
    plt.colorbar()

    plt.tight_layout()
    plt.show(block=True)

def plt_save_mask(mask,i):
    f = plt.figure()
    plt.set_cmap('binary')
    plt.imsave(path+str(i)+'.png',np.rot90(mask,2))
    #plt.imshow(mask)
    #plt.savefig(path+str(i)+'.svg',format='svg')
    #plt.show()

def plt_save_img(mask,i):

    f = plt.figure()
    plt.set_cmap('binary')
    plt.imsave(Path+str(i)+'.png',np.rot90(mask,2))
    #plt.imshow(mask)
    #plt.savefig(path+str(i)+'.svg',format='svg')
    #plt.show()





def show_me_img(path):
  #tif=TIFF.open(path)
  #image=tif.read_image()
  image = cv.imread(path)
  plt.imshow(image,interpolation='nearest')
  plt.show()


def show_me_folder(path):
  for file in sorted(os.listdir(path)):
    base_direc=path+'/'
    print(file)
    tif=TIFF.open(base_direc+file)
    image=tif.read_image()
    plt.imshow(image,interpolation='nearest')
    plt.show()


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))


    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)


if __name__ == '__main__':
    path1 = r'C:\Users\xga\Desktop\example\images\1.png'
    IMG = cv.imread(path1)
    img = cv.imread(path1)
    print(img.shape)
    plot_side_by_side(IMG,img)

