import math
import os
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from random import shuffle

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import *
from utils import *

# Global variable
data_path = 'data/20201229/EXT/resize'
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
         'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
def j_path(parent, child):
    return os.path.join(parent, child)

def data_check(df_name, path):
    data_list = os.listdir(path)
    data_list.sort()
    # raw_list  = [j_path(path,d) for d in data_list if d.endswith('raw')]
    # bmp_list  = [j_path(path,d) for d in data_list if d.endswith('bmp')]
    ext_list  = [j_path(path,d) for d in data_list if d.endswith('.jpg')]
    # res_list  = [j_path(path,d) for d in data_list if d.endswith('RES.bmp')]
    # src_list  = [j_path(path,d) for d in data_list if d.endswith('SRC.bmp')]
    # num_list  = [j_path(path,d) for d in data_list if d.endswith('0.bmp') or d.endswith('1.bmp') \
                # or d.endswith('2.bmp') or d.endswith('3.bmp') or d.endswith('4.bmp')]
    
    # print(f'total: {len(data_list)}')
    # print(f'raw  : {len(raw_list)}')
    # print(f'bmp  : {len(bmp_list)}')
    # print(f'EXT  : {len(ext_list)}')
    # print(f'RES  : {len(res_list)}')
    # print(f'SRC  : {len(src_list)}')
    # print(f'0~4  : {len(num_list)}')

    ext_name = ['EXT','RES','SRC','NUM']
    # _ = [os.mkdir(os.path.join(path,d)) for d in ext_name if not os.path.exists(os.path.join(path,d))]
    # _ = [shutil.copyfile(d,os.path.join(path,ext_name[0],os.path.basename(d))) for d in ext_list]
    # _ = [shutil.copyfile(d,os.path.join(path,ext_name[1],os.path.basename(d))) for d in res_list]
    # _ = [shutil.copyfile(d,os.path.join(path,ext_name[2],os.path.basename(d))) for d in src_list]
    # _ = [shutil.copyfile(d,os.path.join(path,ext_name[3],os.path.basename(d))) for d in num_list]
    print('directories are created, and files have beend copied and moved.')

    # df creation
    assert len(ext_list) > 0, 'wrong image path'
    ext_list = [os.path.basename(d) for d in ext_list]
    ext_list.sort()

    df = pd.DataFrame(ext_list)
    df.to_csv(os.path.join(df_name+'_EXT.csv'))
    df = pd.read_csv(os.path.join(df_name+'_EXT.csv'))
    df.columns   = ['ID','file_name']
    columns_name = ['GT','xmin','ymin','xmax','ymax','ymax_2','note']
    for col in columns_name:
        df[col] = np.nan
    print('raw data:')
    print(df)
    df_name_ext = df_name+'_EXT.csv'
    df.to_csv(df_name_ext, index=False)
    print(f'saved in {df_name_ext}')


def dataframe_creation(df, df_name, data_path, annot_path, resize):
    df = pd.read_csv(df)

    clear_df = df[df['note'] != 'FALSE']
    clear_df = clear_df[clear_df['note'] != 'Weird']
    print('drop FALSE and Weird in note column')
    print(clear_df.info())
    clear_df.drop('note', inplace=True, axis=1)
    clear_df.to_csv(df_name+'_EXT_clear.csv', index=False)

    print('clear data:')
    print(clear_df.info())
    df_name_ext_clear = df_name+'_EXT_clear.csv'
    print(f'saved in {df_name_ext_clear}')

    clear_name = clear_df['file_name'].tolist()
    clear_name = [d[:-4] for d in clear_name]

    xml_path   = os.listdir(annot_path)
    xml_path   = [d for d in xml_path if d.endswith('xml')]
    assert len(xml_path) > 0, 'no annnotation files!'
    xml_path.sort()

    print(f'There are total {len(xml_path)} annotations')
    print(f'There are total {len(clear_name)} clear images')
    xml_path = [xml for xml in xml_path if xml[:-4] in clear_name]
    #assert len(xml_path) == len(clear_name), 'count of the clear images is not match to annotation count, make sure the annotation is correct'    
    print(f'There are {len(xml_path)} annotations in dataframe')
    xml_path = [os.path.join(annot_path,'',d) for d in xml_path if d.endswith('xml')]

    for i, xml in enumerate(xml_path):
        clear_df['GT'].iloc[i], clear_df['xmin'].iloc[i], clear_df['ymin'].iloc[i], \
            clear_df['xmax'].iloc[i], clear_df['ymax'].iloc[i] = xml_reader(xml)
    print('annotation info added')
    print(clear_df.head())

    GT   = clear_df['GT'].tolist()
    GT_1 = [d[:6] for d in GT]
    GT_2 = [d[6:] for d in GT]
    clear_df.drop('GT', inplace=True, axis=1)
    clear_df.insert(2,'GT_1',GT_1,True)
    clear_df.insert(3,'GT_2',GT_2,True)
    print('annotation seperated')
    print(clear_df.head())

    ymin   = clear_df['ymin'].tolist()
    ymax_2 = clear_df['ymax'].tolist()
    ymax_2 = [math.floor((int(d) - int(ymin[i]))/2 + int(ymin[i])) for i, d in enumerate(ymax_2)]
    clear_df['ymax_2'] = ymax_2
    print('annotation bbox seperated')
    print(clear_df.head())

    clear_df.to_csv(df_name + '_EXT_clear_2data.csv', index=False)
    df_name_ext_clear_2data = df_name+'_EXT_clear_2data.csv'
    print(f'saved in {df_name_ext_clear_2data}')

    # train/val/test split
    lpr_dataset_splitting(clear_df, name = df_name + '_EXT_clear_2data', val_ratio=0.3, test_ratio=0.1)
    # image resize
    img_resize(clear_df, data_path=data_path, name = df_name+'_EXT_clear_2data_mode', \
               des_path=os.path.join(data_path,'original_data','resize'), resize=resize)

def xml_reader(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    kind = root.find('object').find('name').text
    loc  = list(root.find('object').find('bndbox'))
    loc  = [d.text for d in loc]
    return kind, loc[0], loc[1], loc[2], loc[3]

def img_resize(df, name, data_path, des_path:str, resize:tuple):
    img_paths = df['file_name'].tolist()
    img_paths = [os.path.join(data_path, 'original_data', d) for d in img_paths]
    
    if not os.path.exists(des_path):
        os.mkdir(des_path)
    # resize the image
    img_resize = [cv.imread(img_path,0) for img_path in img_paths]
    img_resize = [cv.resize(img, resize, \
                  interpolation=cv.INTER_LINEAR) for img in img_resize]
    action = [cv.imwrite(os.path.join(des_path, os.path.basename(img_paths[idk])), img) for idk, img in enumerate(img_resize)]
    assert action[0] is not None, 'des_path cannot be found'

    # resiz the annotation
    xmin, ymin = df['xmin'].tolist(), df['ymin'].tolist()
    xmax, ymax = df['xmax'].tolist(), df['ymax'].tolist()
    ymax2 = df['ymax_2'].tolist()
    xmin = xy_resize(xmin, resize=resize)
    ymin = xy_resize(ymin, resize=resize)
    xmax = xy_resize(xmax, resize=resize)
    ymax = xy_resize(ymax, resize=resize)
    ymax2 = xy_resize(ymax2, resize=resize)

    df['xmin'], df['ymin'] = xmin, ymin
    df['xmax'], df['ymax'] = xmax, ymax
    df['ymax_2'] = ymax2

    df.to_csv(name+'_resize.csv', index=False)
    print('resize done')
    name_resize = name+'_resize.csv'
    print(f'saved in {name_resize}')
    print(df.head())


def xy_resize(xy:list, resize:tuple):
    size = resize[0]
    xy = [int(int(d)*size/1500) for d in xy]
    return xy


def lpr_dataset_splitting(df, name, val_ratio:float, test_ratio:float):
    assert val_ratio + test_ratio < 1, 'the ratio sum cannot >= 1'
    data_len  = len(df.index)
    # 0: train
    # 1: val
    # 2: test
    mode_list = [0]*math.floor(data_len*(1-val_ratio-test_ratio)) + [1]*math.floor(data_len*(val_ratio)) + [2]*math.floor(data_len*(test_ratio))
    
    # to compensate the possible round off
    while len(mode_list) != data_len:
        mode_list.insert(0,0)
    # to produce the same result
    random.seed(10)
    shuffle(mode_list)
    
    df['mode'] = mode_list

    df.to_csv(name+'_mode.csv', index=False)
    print(df.head())
    print('train/val/test mode done')
    name_mode = name+'_mode.csv'
    print(f'saved in {name_mode}')

    return df
   

def pnet_traindata(df, save_dir, which=0):
    '''
    Generating pnet training data
    parameters:
        which: 
            0: only the upper bbox
            1: only the lower bbox
            2: both bboxes
    '''
    assert which in [0,1,2]

    neg_save_dir = os.path.join(save_dir, 'pnet_negative')
    pos_save_dir = os.path.join(save_dir, 'pnet_positive')
    par_save_dir = os.path.join(save_dir, 'pnet_part')
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(par_save_dir):
        os.mkdir(par_save_dir)

    anno_store_path = 'anno_store'
    if not os.path.exists(anno_store_path):
        os.mkdir(anno_store_path)
    fpos = open(os.path.join(anno_store_path, f'pnet_pos.txt'), 'w')
    fneg = open(os.path.join(anno_store_path, f'pnet_neg.txt'), 'w')
    fpar = open(os.path.join(anno_store_path, f'pnet_par.txt'), 'w')
    # the label for later P-net training 
    #  0: neg
    #  1: pos
    # -1: par

    img_paths = df['file_name'].tolist()
    img_paths = [os.path.join(data_path, d) for d in img_paths]
    if which == 0:
        xmin, ymin = df['xmin'].tolist(), df['ymin'].tolist()
        xmax, ymax = df['xmax'].tolist(), df['ymax_2'].tolist()
    elif which == 1:
        xmin, ymin = df['xmin'].tolist(), df['ymax_2'].tolist()
        xmax, ymax = df['xmax'].tolist(), df['ymax'].tolist()
    elif which == 2:
        xmin_1, ymin_1 = df['xmin'].tolist(), df['ymin'].tolist()
        xmax_1, ymax_1 = df['xmax'].tolist(), df['ymax_2'].tolist()
        xmin_2, ymin_2 = df['xmin'].tolist(), df['ymax_2'].tolist()
        xmax_2, ymax_2 = df['xmax'].tolist(), df['ymax'].tolist()

    # index counts
    n_idx = 0 # negative
    p_idx = 0 # positive
    d_idx = 0 # dont care
    for idx, img_path in enumerate(img_paths):
        if which in [0,1]:
            box = np.zeros((1,4), dtype=np.int32)
            box[0,0], box[0,1] = xmin[idx], ymin[idx]
            box[0,2], box[0,3] = xmax[idx], ymax[idx]
        else:
            box = np.zeros((2,4), dtype=np.int32)
            box[0,0], box[0,1] = xmin_1[idx], ymin_1[idx]
            box[0,2], box[0,3] = xmax_1[idx], ymax_1[idx]
            box[1,0], box[1,1] = xmin_2[idx], ymin_2[idx]
            box[1,2], box[1,3] = xmax_2[idx], ymax_2[idx]
        
        img     = cv.imread(img_path,0)
        h, w    = img.shape

        # create negative samples (Iou < 0.3)
        # at least 35 negative samples per image
        neg_num = 0
        while neg_num < 35:
            size_x = np.random.randint(125, w)
            size_y = np.random.randint(20, h / 4)
            nx = np.random.randint(0, w - size_x)
            ny = np.random.randint(0, h - size_y)
            crop_box = np.array([nx, ny, nx + size_x, ny + size_y])

            Iou = IoU(crop_box, box)
            # list of Iou

            cropped_im = img[ny: ny + size_y, nx: nx + size_x]
            ng_resized_im = cv.resize(cropped_im, (47, 12), interpolation=cv.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.bmp" % n_idx)
                fneg.write(save_file + ' 0\n')
                cv.imwrite(save_file, ng_resized_im)
                n_idx += 1
                neg_num += 1
        
        # generate positive and part samples
        # positive IoU >  0.65
        # part     IoU <= 0.4
        bw_l = []
        bh_l = []
        if which in [0,1]:
            bw_l.append(box[0,2] - box[0,0])
            bh_l.append(box[0,3] - box[0,1])
            assert bw_l[0] > 0 and bh_l[0] > 0
        else:
            bw_l.append(box[0,2] - box[0,0])
            bh_l.append(box[0,3] - box[0,1])
            bw_l.append(box[1,2] - box[1,0])
            bh_l.append(box[1,3] - box[1,1])
            assert bw_l[0] > 0 and bh_l[0] > 0 and bw_l[1] > 0 and bh_l[1] > 0

        for _ in range(20):
            for i in range(len(bw_l)):
                bw = bw_l[i]
                bh = bh_l[i]
                size_x = np.random.randint(int(min(bw, bh) * 0.8), np.ceil(1.25 * max(bw, bh)))
                size_y = np.random.randint(int(min(bw, bh) * 0.8), np.ceil(1.25 * max(bw, bh)))

                # delta here is the offset of box center
                delta_x = np.random.randint(-bw * 0.2, bw * 0.2)
                delta_y = np.random.randint(-bh * 0.2, bh * 0.2)

                nx1 = max(box[i,0] + bw / 2 + delta_x - size_x / 2, 0)
                ny1 = max(box[i,1] + bh / 2 + delta_y - size_y / 2, 0)
                nx2 = nx1 + size_x
                ny2 = ny1 + size_y

                # get up those outside the image
                if nx2 > w or ny2 > h:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (box[i,0] - nx1) / float(size_x)
                offset_y1 = (box[i,1] - ny1) / float(size_y)
                offset_x2 = (box[i,2] - nx2) / float(size_x)
                offset_y2 = (box[i,3] - ny2) / float(size_y)

                cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2)]
                resized_im = cv.resize(cropped_im, (47, 12), interpolation=cv.INTER_LINEAR)

                if IoU(crop_box, box[i,...]) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.bmp" % p_idx)
                    fpos.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    pos_resized_im = resized_im
                    cv.imwrite(save_file, pos_resized_im)
                    p_idx += 1
                elif IoU(crop_box, box[i,...]) >= 0.4 and d_idx < 1.2*p_idx + 1:
                    save_file = os.path.join(par_save_dir, "%s.bmp" % d_idx)
                    fpar.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    par_resized_im = resized_im
                    cv.imwrite(save_file, par_resized_im)
                    d_idx += 1

        if idx % 50 == 0: 
            print("%s images done, pos: %s part: %s neg: %s" % (idx+1, p_idx, d_idx, n_idx))
        elif idx == len(img_paths)-1:
            print("%s images done, pos: %s part: %s neg: %s" % (idx+1, p_idx, d_idx, n_idx))

    fneg.close()
    fpar.close()
    fpos.close()

    print('\nExample Images:')
    f, axs = plt.subplots(2,3)
    f.set_figheight(10)
    f.set_figwidth(15)
    axs[0,0].imshow(img)
    axs[0,0].set_title('Original')
    axs[0,0].set_xticks([])
    axs[1,0].imshow(ng_resized_im)
    axs[1,0].set_title('Negative')
    axs[1,0].set_xticks([])
    axs[1,1].imshow(pos_resized_im)
    axs[1,1].set_title('Positive')
    axs[1,1].set_xticks([])
    axs[1,2].imshow(par_resized_im)
    axs[1,2].set_title('Part')
    axs[1,2].set_xticks([])
    f.tight_layout()
    plt.show()

def onet_traindata(df, save_dir, which=0):
    '''
    Generating pnet training data
    parameters:
        which: 
            0: only the upper bbox
            1: only the lower bbox
            2: for both
    '''
    assert which in [0,1,2]
    neg_save_dir = os.path.join(save_dir, 'onet_negative')
    pos_save_dir = os.path.join(save_dir, 'onet_positive')
    par_save_dir = os.path.join(save_dir, 'onet_part')
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(par_save_dir):
        os.mkdir(par_save_dir)

    anno_store_path = 'anno_store'
    if not os.path.exists(anno_store_path):
        os.mkdir(anno_store_path)
    fpos = open(os.path.join(anno_store_path, f'onet_pos.txt'), 'w')
    fneg = open(os.path.join(anno_store_path, f'onet_neg.txt'), 'w')
    fpar = open(os.path.join(anno_store_path, f'onet_par.txt'), 'w')
    # the label for later P-net training 
    #  0: neg
    #  1: pos
    # -1: par
    
    # onet sample output size
    image_size = (94, 24)

    img_paths = df['file_name'].tolist()
    img_paths = [os.path.join(data_path, d) for d in img_paths]
    if which == 0:
        xmin, ymin = df['xmin'].tolist(), df['ymin'].tolist()
        xmax, ymax = df['xmax'].tolist(), df['ymax_2'].tolist()
    elif which == 1:
        xmin, ymin = df['xmin'].tolist(), df['ymax_2'].tolist()
        xmax, ymax = df['xmax'].tolist(), df['ymax'].tolist()
    elif which == 2:
        xmin_1, ymin_1 = df['xmin'].tolist(), df['ymin'].tolist()
        xmax_1, ymax_1 = df['xmax'].tolist(), df['ymax_2'].tolist()
        xmin_2, ymin_2 = df['xmin'].tolist(), df['ymax_2'].tolist()
        xmax_2, ymax_2 = df['xmax'].tolist(), df['ymax'].tolist()

    # need device since need to pass the Pnet
    device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_path = 'weights/pnet/pnet_best.pth'
    # index counts
    n_idx = 0 # negative
    p_idx = 0 # positive
    d_idx = 0 # dont care
    for idx, img_path in enumerate(img_paths):
        if which in [0,1]:
            box = np.zeros((1,4), dtype=np.int32)
            box[0,0], box[0,1] = xmin[idx], ymin[idx]
            box[0,2], box[0,3] = xmax[idx], ymax[idx]
        else:
            box = np.zeros((2,4), dtype=np.int32)
            box[0,0], box[0,1] = xmin_1[idx], ymin_1[idx]
            box[0,2], box[0,3] = xmax_1[idx], ymax_1[idx]
            box[1,0], box[1,1] = xmin_2[idx], ymin_2[idx]
            box[1,2], box[1,3] = xmax_2[idx], ymax_2[idx]
        
        img   = cv.imread(img_path,0)
        image = img.copy()
        bboxes = create_mtcnn_net(image, [50,50], device, p_model_path=weight_path)
        dets = np.round(bboxes[:, 0:4])
        if dets.shape[0] == 0:
            continue
        
        for det in dets:
            x_left, y_top, x_right, y_bottom = det[0:4].astype(int)
            # NOTE: slightly increase the y
            y_top    -= 5
            y_bottom += 5
            det_width  = x_right - x_left + 1
            det_height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if det_width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 \
                or y_bottom > img.shape[0] - 1 or y_bottom-y_top < 5:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(det, box)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1]
            try:
                resized_im = cv.resize(cropped_im, image_size, interpolation=cv.INTER_LINEAR)
            except:
                print(cropped_im.shape)
                print(y_top)
                print(y_bottom + 1)
                raise

            # save negative images and write label
            # IoU with all gts must below 0.3
            if np.max(Iou) < 0.3 and n_idx < 3.2*p_idx+1:
                save_file = os.path.join(neg_save_dir, "%s.bmp" % n_idx)
                fneg.write(save_file + ' 0\n')
                cv.imwrite(save_file, resized_im)
                ng_resized_im = resized_im

                n_idx += 1
            else:
                # find gt_box with the highest iou
                idx_Iou         = np.argmax(Iou)
                assigned_gt     = box[idx_Iou]
                x1, y1, x2, y2  = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(det_width)
                offset_y1 = (y1 - y_top) / float(det_height)
                offset_x2 = (x2 - x_right) / float(det_width)
                offset_y2 = (y2 - y_bottom) / float(det_height)


                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.bmp" % p_idx)
                    fpos.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv.imwrite(save_file, resized_im)
                    pos_resized_im = resized_im
                    
                    p_idx += 1

                elif np.max(Iou) >= 0.4 and d_idx < 1.2*p_idx + 1:
                    save_file = os.path.join(par_save_dir, "%s.bmp" % d_idx)
                    fpar.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv.imwrite(save_file, resized_im)
                    par_resized_im = resized_im

                    d_idx += 1

        if idx % 50 == 0: 
            print("%s images done, pos: %s part: %s neg: %s" % (idx+1, p_idx, d_idx, n_idx))
        elif idx == len(img_paths)-1:
            print("%s images done, pos: %s part: %s neg: %s" % (idx+1, p_idx, d_idx, n_idx))

    fneg.close()
    fpar.close()
    fpos.close()

    print('\nExample Images:')
    f, axs = plt.subplots(2,3)
    f.set_figheight(10)
    f.set_figwidth(15)
    axs[0,0].imshow(img)
    axs[0,0].set_title('Original')
    axs[0,0].set_xticks([])
    axs[1,0].imshow(ng_resized_im)
    axs[1,0].set_title('Negative')
    axs[1,0].set_xticks([])
    axs[1,1].imshow(pos_resized_im)
    axs[1,1].set_title('Positive')
    axs[1,1].set_xticks([])
    axs[1,2].imshow(par_resized_im)
    axs[1,2].set_title('Part')
    axs[1,2].set_xticks([])
    f.tight_layout()
    plt.show()

def assemble_split(dir, output_file, net_name:str,val_ratio:float):
    '''
    Assemble the pnet pos/neg/par .txt and split them into train/val
    '''
    net_name = net_name.lower()
    _, file_extension = os.path.splitext(output_file)
    assert file_extension == '', 'output_file should not contain extension'
    assert val_ratio < 1 and val_ratio > 0
    train_output = output_file+'_train.txt'
    val_output   = output_file+'_val.txt'
    if os.path.exists(train_output):
        os.remove(train_output)
    if os.path.exists(val_output):
        os.remove(val_output)


    paths = os.listdir(dir)
    paths = [os.path.join(dir, d) for d in paths if d.startswith(f'{net_name}')]
    total_lines = 0
    t_lines     = 0
    v_lines     = 0
    for path in paths:
        with open(path, 'r') as f:
            lines        = f.readlines()
        total_lines += len(lines)
        shuffle(lines) # shuffle the list inplace
        train_lines = lines[:int(len(lines) * (1-val_ratio))]
        val_lines   = lines[int(len(lines) * (1-val_ratio)):]
        t_lines += len(train_lines)
        v_lines += len(val_lines)

        with open(train_output, 'a+') as f:
            _ = [f.write(d) for d in train_lines]
        with open(val_output, 'a+') as f:
            _ = [f.write(d) for d in val_lines]

    print(f'Total {total_lines} samples')
    print(f'{t_lines} training, {v_lines} validation')
    print(f'Image list is output to {output_file}')


def preprocess(img):
    """Preprocessing step before feeding the network.

    Arguments:
        img: a float numpy array of shape [h, w].

    Returns:
        a float numpy array of shape [1, 1, h, w].
    """
    img = np.asarray(img, 'float32')
    img = (img - 37.7) / 255.
    # channel
    img = np.expand_dims(img, 0)
    # batch
    img = np.expand_dims(img, 0)
    return img

def normalize_img(df):
    img_paths = df['file_name'].tolist()
    img_paths = [os.path.join(data_path, d) for d in img_paths]
    
    img       = [cv.imread(d,0) for d in img_paths]
    mean_list = [np.mean(d) for d in img]
    mean      = np.sum(mean_list)/len(mean_list)
    return mean


def denoiser(img_path, des_path, kernel=3):
    if not os.path.exists(des_path):
        os.mkdir(des_path)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel,kernel))

    img_paths = os.listdir(img_path)
    img_paths = [d for d in img_paths if d.endswith('bmp')]
    img_paths = [os.path.join(img_path, d) for d in img_paths]
    
    imgs = [cv2.imread(d,0).astype(np.uint8) for d in img_paths]
    imgs = [cv2.morphologyEx(d, cv2.MORPH_CLOSE, kernel) for d in imgs]
    imgs = [cv2.morphologyEx(d, cv2.MORPH_CLOSE, kernel) for d in imgs]
    action = [cv2.imwrite(os.path.join(des_path, os.path.basename(img_paths[i])), d) for i, d in enumerate(imgs)]

    print('Denoising done')



if __name__ == "__main__":
    # ---- Data Preprocessing ----
    # data check and mkdir
    data_check(df_name='20210801',path='dataall/Digits')
    # df creation and preprocessing
    dataframe_creation(df='20210801_EXT.csv', df_name='20210801', data_path='dataall/Digits', \
                       annot_path='dataall/digitsxml', resize = (750,750))

    # ---- Data Generating ----
    # df = pd.read_csv('20201229_EXT_clear_2data_mode_resize.csv')
    # Generating Pnet samples
    # pnet_traindata(df,  'data/20201229/EXT/resize', which=2)
    # Preparing Pnet samples (also spliting train/val)
    # assemble_split('anno_store', output_file='anno_store/pnet_imglist', net_name='pnet', val_ratio=0.3)

    # pnet training
    # os.system('python train_pnet.py')

    # Generating Onet samples
    # DANGER: Make sure you have trained the Pnet
    # onet_traindata(df, 'data/20201229/EXT/resize', which=2)

    # Preparing Onet samples (also spliting train/val)
    # assemble_split('anno_store', output_file='anno_store/onet_imglist', net_name='onet',  val_ratio=0.3)

    # os.system('python train_onet.py')

    # map
    # os.system('python mtcnn_metric.py')

