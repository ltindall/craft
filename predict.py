"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft.craft_utils as craft_utils
import craft.imgproc as imgproc
#import craft.file_utils as file_utils
import json
import zipfile

from craft.craft_model import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")



def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size, mag_ratio, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    #if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def load_predict_net(trained_model='craft/weights/craft_mlt_25k.pth'): 
    # load net
    cuda= torch.cuda.is_available()
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + trained_model + ')')
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()
    return net

def predict_seg(
    image,
    trained_model='craft/weights/craft_mlt_25k.pth',
    text_threshold=0.7,
    low_text=0.4,
    link_threshold=0.4,
    canvas_size=3200,
    mag_ratio=1.5,
    poly=False,
    show_time=False,
    #test_folder='craft/data/',
    refine=False,
    refiner_model='craft/weights/craft_refiner_CTW1500.pth', 
    preload_net=None
):
    cuda= torch.cuda.is_available()
    # Load CRAFT model with trained weights 
    if preload_net is not None: 
        net = preload_net
    else: 
        net = CRAFT()   
        #print('Loading weights from checkpoint (' + trained_model + ')')
        if cuda:
            net.load_state_dict(copyStateDict(torch.load(trained_model)))
        else:
            net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))
        if cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False
        net.eval()

    # LinkRefiner
    refine_net = None
    if refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        #print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
        if cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))
        refine_net.eval()
        poly = True
    t = time.time()
    print("finding text bounding boxes...")
    bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size, mag_ratio, refine_net)
    print("elapsed time : {}s".format(time.time() - t))
    # bboxes is an array with shape (n, 4, 2); each bbox is orderdered [bottom left, bottom right, top right, top left]
    return bboxes

def get_bbox_centers(bboxes): 
    centers = []
    for box in bboxes: 
        x = int(box[1][0] - (box[1][0] - box[0][0])/2)
        y = int(box[3][1] - (box[3][1] - box[0][1])/2)
        centers.append([x,y])

    return np.array(centers)

def nearest_neighbor_pairs(centers, timestamp_img=None): 
    ''' 
    Returns the closest neighbor pairs. 
    '''
    center_times = np.zeros(len(centers))
    if timestamp_img is not None: 
        for i, center in enumerate(centers): 
            center_times[i] = timestamp_img[tuple(center[::-1])]
    pairs = []
    for i, centerA in enumerate(centers): 
        min_dist = None 
        min_pair = None
        min_center_idx = None
        for j, centerB in enumerate(centers): 
            if i != j: 
                distance = np.linalg.norm(centerA - centerB) 
                if min_dist == None or min_dist > distance: 
                    min_dist = distance
                    min_center_idx = j
        if center_times[i] == center_times[min_center_idx]: 
            min_pair = np.array([centerA, centers[min_center_idx]])
            pairs.append(min_pair)
        #if min_pair is not None: 
        #    pairs.append(min_pair)
    return np.array(pairs)

def cluster_centers_on_location(centers): 
    '''
    Returns pairs where the distance between them is less than 0.5*average distance between all pairs. 
    '''
    distances = []
    for i, centerA in enumerate(centers): 
        for j, centerB in enumerate(centers[i+1:]): 
            distances.append(np.linalg.norm(centerA - centerB))
    distances = np.array(distances)
    mean_distance = np.mean(distances)
    pairs = []
    for i, centerA in enumerate(centers): 
        for j, centerB in enumerate(centers[i+1:]): 
            if(np.linalg.norm(centerA - centerB) < 0.5 * mean_distance): 
                pairs.append(np.array([centerA, centerB]))
    return np.array(pairs)

def cluster_centers_on_time(centers, timestamp_img): 
    center_times = []
    for center in centers: 
        center_times.append(timestamp_img[tuple(center[::-1])])
    time_clusters = []
    for c_time in np.unique(center_times): 
        c_time_centers = centers[center_times==c_time]
        time_cluster = []
        for i, centerA in enumerate(c_time_centers): 
            for j, centerB in enumerate(c_time_centers[i+1:]): 
                time_cluster.append(np.array([centerA, centerB]))
        time_clusters.append(np.array(time_cluster))
    return np.array(time_clusters)

