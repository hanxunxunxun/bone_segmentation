# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:46:08 2019

@author: Rex Zhu
"""

import cv2
import numpy as np
from tempfile import TemporaryFile
import glob
import os
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.model_selection import train_test_split
from skimage import transform
from skimage import io
from tqdm import tqdm
#from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim


#WSI_PATH = 'E:\ACSE\Project\Bones Segmentation\Fast Edge detection\Results\M9_Bone_Labels'
#paths = glob.glob(os.path.join(WSI_PATH,'*.jpg'))
#paths.sort()
#img = np.array(np.zeros(shape=[len(paths),36,36,1]));
#
#n=0
#for path in paths:
#    raw = cv2.imread(path)
#    raw1= raw[:,:,1]
#    for x in range(1440):
#        for y in range(1440):
#            if raw1[x,y]<250:
#                raw1[x,y]=0
#            else:
#                raw1[x,y]=raw1[x,y]
#        
#        
#    i = transform.rescale(raw1,0.025,anti_aliasing=False)   
#    img[n, :, :, 0]= i[:,:]
#    n=n+1    
#    print(n)    
#        
#        
#np.save("M9_l.npy",img)       
        
        
        
        
        
#WSI_PATH ='E:\ACSE\Project\Bones Segmentation\Fast Edge detection\Data\No tumour (original dataset)-20190926T133211Z-001\No tumour (original dataset)'
#WSI_PATH2 = 'E:\ACSE\Project\Bones Segmentation\Fast Edge detection\Data\No tumour (with artificial lesion)-20190923T125735Z-001\No tumour (with artificial lesion)'
WSI_PATH = r'/Users/hanxun/Desktop/数据集/No\ tumour\ \(original\ dataset\)'
WSI_PATH2 = r'/Users/hanxun/Desktop/数据集/No\ tumour\ \(with\ artificial\ lesion\)'

paths1 = glob.glob(os.path.join(WSI_PATH,'*.bmp'))
paths1.sort()
paths2 = glob.glob(os.path.join(WSI_PATH2,'*.bmp'))
paths2.sort()
ori = np.array(np.zeros(shape=[len(paths1),10,10,1]))
le = np.array(np.zeros(shape=[len(paths2),10,10,1]))
label = np.array(np.zeros(shape=[len(paths1),1440,1440,1]))
label1 = np.array(np.zeros(shape=[len(paths1),22,26,1]))
#n=0   
#for path in tqdm(range(len(paths1))):
#    original = cv2.imread(paths1[path])
#    original1 = original[:,:,1]
#    lesion = cv2.imread(paths2[path])
#    lesion1 = lesion[:,:,1]
#    
#    for x in range(898):
#        for y in range(1043):
#            if original1[x,y]>lesion1[x,y]:
#                label[path,x,y,0] = 500
#            else:
#                label[path,x,y,0] = lesion1[x,y]
#    
#    
#for n in tqdm(range(len(paths1))):
#    l[n,:,:,0] = transform.rescale(label[n,:,:,0],0.025,anti_aliasing=False) 



for path in tqdm(range(len(paths1))):
    original = cv2.imread(paths1[path])
    original1 = original[:,:,1]
    lesion = cv2.imread(paths2[path])
    lesion1 = lesion[:,:,1]
    
    ori = transform.rescale(original1,0.025,anti_aliasing=False) 
    le = transform.rescale(lesion1,0.025,anti_aliasing=False) 
    
    for x in range(len(ori[:,1])):
        for y in range(len(ori[1,:])):
            if ori[x,y]<le[x,y]:
                label1[path,x,y,0] = 10
            else:
                label1[path,x,y,0] = ori[x,y]
                
                
img =  np.array(np.zeros(shape=[len(paths1),22,26,1]))   
for path in tqdm(range(len(paths1))):
    original = cv2.imread(paths1[path])
    original1 = original[:,:,1]
    img[path,:,:,0] = transform.rescale(original1,0.025,anti_aliasing=False) 
    
    np.save("MML.npy",label1)





  
#np.save("MML_l.npy",label)  
#
#
#
#
#
#
#a=label1[496,:,:,0]
#b=img[496,:,:,0]
#cv2.imshow("1",transform.rescale(b,40,anti_aliasing=False))
#cv2.waitKey()

np.save("MML_l.npy",label1)

#
#cv2.imshow("1",transform.rescale(ori,40,anti_aliasing=False))
#cv2.waitKey()


########################################################################## Make Bone Data ##############################################################################################################################
WSI_PATH = '/Users/hanxun/Desktop/数据集/M9/image'

paths1 = glob.glob(os.path.join(WSI_PATH,'*.jpg'))
paths1.sort()

im = np.array(np.zeros(shape=[len(paths1),90,90,1]))
label = np.array(np.zeros(shape=[len(paths1),90,90,1]))

for path in tqdm(range(len(paths1))):
    original = cv2.imread(paths1[path])
    original1 = original[:,:,1]
    (thresh, bw) = cv2.threshold(original1, 127, 255, cv2.THRESH_BINARY)
    im[path,:,:,0] = transform.rescale(original1,1/16,anti_aliasing=False) 
    label[path,:,:,0] = transform.rescale(bw,1/16,anti_aliasing=False)

dataset_image_train, dataset_image_eval, train_l, eval_l =  train_test_split(im,label,test_size=0.3,random_state=0)

np.save("M90_train.npy",dataset_image_train)
np.save("M90_eval.npy",dataset_image_eval)
np.save("M90_tl.npy",train_l)
np.save("M90_el.npy",eval_l)

#################################################################################################################################################################
WSI_PATH = '/Users/hanxun/Desktop/数据集/M9/image'

paths1 = glob.glob(os.path.join(WSI_PATH,'*.jpg'))
paths1.sort()

im = np.array(np.zeros(shape=[len(paths1),27*27]))
label = np.array(np.zeros(shape=[len(paths1),27*27]))

for path in tqdm(range(len(paths1))):
    original = cv2.imread(paths1[path])
    original1 = transform.rescale(original[:,:,1],1/53,anti_aliasing=False) 
    (thresh, bw) = cv2.threshold(original1, 127, 255, cv2.THRESH_BINARY)
    im[path,:] = original1.flatten()
    bw2=transform.rescale(bw,1/53,anti_aliasing=False)
    label[path,:] = bw2.flatten()

dataset_image_train, dataset_image_eval, train_l, eval_l =  train_test_split(im,label,test_size=0.3,random_state=0)

dataset_image_test, dataset_image_eval, test_l, eval_l =  train_test_split(im,label,test_size=0.3,random_state=0)

np.save("M27_train.npy",dataset_image_train)
np.save("M27_eval.npy",dataset_image_eval)
np.save("M27_tl.npy",train_l)
np.save("M27_el.npy",eval_l)
np.save("M27_testl.npy",test_l)
np.save("M27_test.npy",dataset_image_test)









########################################################################## Downsample error ##############################################################################################################################
       
WSI_PATH = '/Users/hanxun/Desktop/数据集/M9/label'
paths1 = glob.glob(os.path.join(WSI_PATH,'*.jpg'))
paths1.sort()
down = np.array(np.zeros(shape=[len(paths1),27,27,1]))
original1 = np.array(np.zeros(shape=[len(paths1),1440,1440,1]))
for path in tqdm(range(len(paths1))):
    original= cv2.imread(paths1[path])
    original1[path,:,:,0] = original[:,:,0]

for path in tqdm(range(len(paths1))):
    down[path,:,:,0] =  transform.rescale(original1[path,:,:,0],1/53,anti_aliasing=False) 

up = np.array(np.zeros(shape=[len(paths1),1440,1440,1]))
for path in tqdm(range(len(paths1))):
    up[path,:,:,0] =  transform.rescale(down[path,:,:,0],53,anti_aliasing=False) 
    
    
# Calculation of parameters
    # means:
    
ssim_down = np.array(np.zeros(shape=[len(paths1)]))    
for path in tqdm(range(len(paths1))):
    ssim_down[path] = ssim(original1[path,:,:,0],up[path,:,:,0] , data_range=up[path,:,:,0].max() - up[path,:,:,0].min())

ssim_down = ssim_down[~np.isnan(ssim_down)]    
ssim_mean = np.mean(ssim_down)








cv2.imshow("1",transform.rescale(label[1200,:,:,0],40,anti_aliasing=False) )
cv2.waitKey()


cv2.imshow("2",original1[1200,:,:,0])
cv2.waitKey()


#############################################################################################################


import pandas as pd
import numpy as np

truth=np.load('M27_el.npy')
t = np.array(np.zeros(shape=[371,27*27]))
for n in range(371):
    t[n,:]= truth[n,:,:,0].flatten()


df_t = pd.DataFrame()
df_t = df_t.append(pd.DataFrame(t))
df_t.to_csv('Truth//truth.csv', header=None, index=None)


























