#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 22:14:29 2017

@author: michelle
"""
import numpy as np
import nibabel as nib
def precision_recall_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)

    precision_c = []
    recall_c = []
    for c in range(len(c_list)):
        move_img_c = (move_img == c_list[c]) # pred
        refer_img_c = (refer_img == c_list[c])# gd
        # intersection
        ints = np.sum(np.logical_and(move_img_c, refer_img_c)*1)
        # precision
        prec = ints / (np.sum(move_img_c*1) + 0.001)
        # recall
        recall = ints / (np.sum(refer_img_c*1) + 0.001)

        precision_c.append(prec)
        recall_c.append(recall)

    return precision_c, recall_c

def precision_recall_n_class1(gd_img, pred_img):
    # list of classes
    #c_list = np.unique(gd_img)

    gd_img_c = (gd_img == 0) # pred
    pred_img_c = (pred_img == 0)# gd

    # intersection
    ints = np.sum(np.logical_and(gd_img_c, pred_img_c)*1)
    # precision
    prec = ints / (np.sum(gd_img_c*1) + 0.001)
    # recall
    recall = ints / (np.sum(pred_img_c*1) + 0.001)

    return prec, recall

if __name__ == '__main__':
    img_path ='/media/269G/dataset_origin/huaxiyiyuan_transform/pred/prediction_normalized_ranjifang.npz/ranjinfang.nii.gz'
    lab_path = '/media/269G/dataset_origin/huaxiyiyuan_transform/groudtruth/ranjifang.nii'
    move_img = nib.load(img_path).get_data().copy()
    refer_img = nib.load(lab_path).get_data().copy()
    #a = np.sum(np.logical_and([True,False],[True,False])*1)
    #print (a)
    refer_img[refer_img>0] = 1
    precision_recall_n_class1(move_img,refer_img)
    
    