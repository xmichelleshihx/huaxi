#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 05:33:38 2017

@author: michelle
"""
import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import math
class preprocessing:
    #sliceThickness = 1
def __init__(self,
        sliceNumber = 513
        SeriesNumber = 13
def preprocessing():
    # get the folder_name 
    #    data_path = '/home/michelle/dataset/huaxiyixian/data/'
    #    file_names_temp = os.listdir(data_path)
    #    file_names = [file_names_temp[0]+'2015-11-2linlin/BJHIYHVB/DICOM/20151104/17380001',
    #                  file_names_temp[1]+'2015-8-18lizhongsheng0009565423/DICOM/20150818/18450001',
    #                  file_names_temp[2]+'2015-8-14liuhuaixiang0015754628/DICOM/20150815/13080002',
    #                  file_names_temp[3]+'2015-10-13huxing/KJGVIK/DICOM/20151015/21180001',
    #                  file_names_temp[4]+'2015-8-6ranjifang0015709990/DICOM/20150806/20450001']
    #    print(file_names)
    
    # read the pydicom data as a nii
    # change the dicom root folder path
    data_path = '/home/michelle/dataset/huaxiyiyuan/data/'   
    # change the dicom folder path 
    file_names = ['ranjifang/DICOM/20150806/20450001']
    #'linlin/BJHIYHVB/DICOM/20151104/17380001'
    #'lizhongsheng/DICOM/20150818/18450001'
    #'liuhuaixiang/DICOM/20150815/13080002'
    #'huxing/KJGVIK/DICOM/20151015/21180001'
    #'ranjifang/DICOM/20150806/20450001'
    #'hewenjun'
    #'kangyouxian/DE_#PP_DE_ABDOMEN_1_0_D30F_A_100KV_0012'
    #'wangjia/20580001'
    #'fuqiong/DICOM/20150813/22090001'
    initial_list = []
    data = {'file_names':file_names,'data':initial_list,'label':initial_list}
    for name in file_names:       
        #data[name] = get_pydicom(data_path, name)
        data_temp = get_pydicom(data_path,name)
        print("end")
#        data['data'] = data['data'].append(data_temp)
#        data['label'] = data['label'].append(data_temp)

def get_pydicom(data_path,name):
    # read the pydicom data as a nii
    # change the groudtruth path#
    groudtruth_path = '/home/michelle/dataset/huaxiyiyuan/groudtruth/fuqiong.nii'
    linlin = nib.load(groudtruth_path)
    affine = linlin.affine
    #linlin.
    
    path = data_path+name
    lstFilesDCM = []
    print(path)
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
    # Get ref file
    RefDs = pydicom.read_file(lstFilesDCM[0])    
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    # change the third paramter, which is the slice of the volume
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), sliceNumber)    
    
    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    #ArrayDicom1 = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    # loop through all the DICOM files
    count = 0
    slice_names = []
    index_box = []
    
    # read the file in a xxx/path, which is dicom
    for filenameDCM in lstFilesDCM:
    # read the file
        slice_names.append(filenameDCM)
        ds_temp = pydicom.read_file(filenameDCM)
        #format(ds_temp.shape[0],ds_temp.shape[1])
        if ds_temp.SeriesNumber==13:
            ds = ds_temp    
            # store the raw image data               
            index_box.append(int(ds.InstanceNumber)-1)
            print(int(ds.InstanceNumber)-1)
            ArrayDicom[:, :, int(ds.InstanceNumber)-1] = ds.pixel_array                             
            count = count + 1
            print(count)
    # rotation
    ArrayDicom = np.rot90(ArrayDicom, 1, (2,1))
    ArrayDicom = np.rot90(ArrayDicom, -1, (1,0))
    ArrayDicom = np.rot90(ArrayDicom, -1, (2,0))
    img = nib.Nifti1Image(ArrayDicom,affine)   
    # change the save file name#
    nib.save(img, 'xxx.nii')
    # set a debug here#
    print("hello")
    
if __name__ == '__main__':
    preprocessing()