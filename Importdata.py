# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:21:26 2019

@author: Rex Zhu
"""

import numpy as np
import tensorflow as tf
import tqdm
import pickle_utils as pu
from gpflow import settings
import sys
import pandas as pd
import os
import gpflow
import xlrd
import csv

def main(_):
    train_path = 'E:/ACSE/Project/Traffic Gaussian/Codes/ConvnetsAsGP/Data/train.csv'
    validation_path = 'E:/ACSE/Project/Traffic Gaussian/Codes/ConvnetsAsGP/Data/validaiton.csv'
    test_path = 'E:/ACSE/Project/Traffic Gaussian/Codes/ConvnetsAsGP/Data/test.csv'
    trainlabel_path = 'E:/ACSE/Project/Traffic Gaussian/Codes/ConvnetsAsGP/Data/trainlabel.csv'
    validationlabel_path = 'E:/ACSE/Project/Traffic Gaussian/Codes/ConvnetsAsGP/Data/validaitonlabel.csv'
    testlabel_path = 'E:/ACSE/Project/Traffic Gaussian/Codes/ConvnetsAsGP/Data/testlabel.csv'
    
    print("Traffic training data loaded from:", train_path)
    trainset_read = pd.read_csv(train_path, sep=',')
#    trainset=trainset_read.values
    trainset=np.float64(trainset_read)
    trainlable_read = pd.read_csv(trainlabel_path, sep=',')
    trainlable=np.float64(trainlable_read)
    
    print("Traffic training data loaded from:", train_path)
    validationset_read = pd.read_csv(validation_path, sep=',')
#    validationset=validationset_read.values
    validationset=np.float64(validationset_read)
    validationlable_read = pd.read_csv(validationlabel_path, sep=',')
    validationlable=np.float64(validationlable_read)
    
    print("Traffic training data loaded from:", train_path)
    testset_read = pd.read_csv(test_path, sep=',')
#    testset=testset_read.values
    testset=np.float64(testset_read)
    testlable_read = pd.read_csv(testlabel_path, sep=',')
    testlable=np.float64(testlable_read)
       


    
    r = tuple(a.astype(settings.float_type) for a in [
        trainset, trainlable,
        validationset, validationlable,
        testset, testlable])
    return r
    
    
