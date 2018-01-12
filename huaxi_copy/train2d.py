"""
debug script for 2d !!!! meow!
"""
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import unet2dpls_4_mask_huaxi as u2d
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
#import pymedimage.visualize as viz
import os
import sys
import logging
import pdb
import datetime
import argparse

logging.basicConfig(filename = "general_log", level = logging.DEBUG)

test_flag = False # True for test
currtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
train_dir= "/media/269G/dataset_origin/huaxiyiyuan_transform/train"
label_dir= "/media/269G/dataset_origin/huaxiyiyuan_transform/groudtruth"
val_dir= "/media/269G/dataset_origin/huaxiyiyuan_transform/val"
test_dir= "/media/269G/dataset_origin/huaxiyiyuan_transform/train"
#output_path = "/home/oycheng/Documents/project/result2d/" + currtime + "huaxi_new"
output_path = "/media/269G/dataset_origin/huaxiyiyuan_transform/output"
mask_path = output_path + "/masks/"
num_cls = 2
verbose = False
batch_size = 5 #I want to call the police since even 2 will explodes the memory
if test_flag is True:
    batch_size = 1
reg_coe = 1e-4
learning_rate = 1e-3
#used to be 0.001
training_iters = 10
epochs = 50000
checkpoint_space = 3000
image_summeris = True
restore = True
restored_path = output_path
aux_flag = True
miu_cross = 1.0 # 100.0
miu_dice = 1.0
miu_aux1 = 0.0
miu_aux2 = 0.0
miu_aux3 = 0.0
optimizer = 'adam'
miu_cross *= 1.
miu_dice *= 1.
lbd_fp = 1.0
lbd_p = 1.0

cost_kwargs = {
    "regularizer": reg_coe,
    "miu_cross": miu_cross,
    "miu_dice": miu_dice,
    "dice_flag": True,
    "cross_flag": True,
    "aux_flag": aux_flag,
    "miu_aux1": miu_aux1,
    "miu_aux2": miu_aux2,
    "miu_aux3": miu_aux3,
    "lbd_fp": lbd_fp,
    "lbd_p": lbd_p,
}

opt_kwargs = {
    "learning_rate": learning_rate
}

weight_dict = {
"0": 0.003145286348454345,
"5": 5.1815551943721845,
"2": 2.7089654324818744,
"7": 0.32727025429249484,
"1": 0.31043663665548976,
"6": 0.14430344081928984,
"3": 1.0003337689657856,
"4": 1.0492824193859533
}

#pdb.set_trace()
class_weights = [value for key, value in sorted(weight_dict.items(), key = lambda x: int(x[0]))]
cost_kwargs["class_weights"] = np.array(class_weights)
#pdb.set_trace()
'''
contour_map = {
    "Background": 0,
    "Bowel": 1,
    "Duodenum": 2,
    "L-kidney": 3,
    "R-kidney": 4,
    "Spinal_Cord": 5,
    "Liver": 6,
    "Stomach":7
}
contour_map = {
    "Background": 0,
    "CTV": 1
}
'''
contour_map = { 'Background': 0,
                'CTV1': 1,
               'Bladder': 2,
               'Bome_Marrow': 3,
               'Femoral_Head_L': 4,
               'Femoral_Head_R': 5,
               'Rectum': 6,
               'Small_Intestine': 7
               }

def _read_lists(fid):
    """ read train list and test list """
    if not os.path.isfile(fid):
        return None
    with open(fid,'r') as fd:
        _list = fd.readlines()

    my_list = []
    for _item in _list:
        if len(_item) < 5:
            _list.remove(_item)
        my_list.append(_item.split('\n')[0])
    return my_list

def main(lr_update_flag):
#    train_list_dir = os.listdir()
#    train_list = _read_lists(train_fid)
#    val_list = _read_lists(val_fid)
#    test_list = _read_lists(test_fid)
    pdb.set_trace()
    if test_flag is False:
        try:
            os.makedirs(output_path)
            os.makedirs(mask_path)
        except:
            print("folder exist!")

    if verbose:
        print("Start building the data generator...")
    #pdb.set_trace()
    my_unet3d = u2d.Unet(channels = 3, batch_size = batch_size,  n_class = num_cls, image_summeris = image_summeris, test_flag = test_flag, cost_kwargs = cost_kwargs)
    if verbose:
        print("Network as been built!!!!!")
    my_trainer = u2d.Trainer(my_unet3d, train_dir, label_dir, val_dir, test_dir, mask_folder = mask_path, num_cls = num_cls, \
                             batch_size = batch_size, opt_kwargs = opt_kwargs, checkpoint_space = checkpoint_space,\
                             optimizer = optimizer, lr_update_flag = lr_update_flag)
    # start tensorboard before getting started
#    if restore is True:
#        output_path = restored_pth
    command2 = "fuser 6006/tcp -k"
    os.system(command2)
    command1 = "tensorboard --logdir=" + output_path + " --port=6006 " +  " &"
    os.system(command1)

    print("Now start training...")
    if test_flag is True:
        my_trainer.test(output_path = output_path, restored_path = restored_path)
        exit()
    if restore is True:
        my_trainer.train(output_path = output_path, restored_path = restored_path, training_iters = training_iters, epochs = epochs, restore = True)
    else:
        my_trainer.train(output_path = output_path, training_iters = training_iters, epochs = epochs )
    #my_trainer._test_ppl()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_update_flag", action = "store_true", default = False)
    args = parser.parse_args()
    lr_update_flag = args.lr_update_flag
    main(lr_update_flag)

