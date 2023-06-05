import numpy as np
import os
import tqdm
import cv2
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import matplotlib.pyplot as plt

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import _PanopticPrediction

from .regression import DummyData, PolynomialModel, train_step

class DepthPredictor():
    def __init__(self, cfg, args, logger, dmax = 255.):
        self.cfg = cfg
        self.args = args
        self.logger = logger
        self.dmax = dmax

        self.device = torch.device('cpu')
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
        self.stuff_list = self.metadata.stuff_classes

    def predict(self, image) :
        self.height, self.width, _ = image.shape

        predictions = self.predictor(image)
        panoptic_seg, segments_info = predictions["panoptic_seg"]
        pred = _PanopticPrediction(panoptic_seg.to('cpu'), segments_info, self.metadata)

        all_stuffs = list(pred.semantic_masks())
        if len(all_stuffs) == 0 : return image
        Semanticmasks, Semanticinfos = list(zip(*all_stuffs))

        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0 : return image
        Instancemasks, Instanceinfos = list(zip(*all_instances))

        grounds, sky, nongrounds = self.findGround_Sky(Semanticmasks, Semanticinfos)
        if len(grounds) == 0 : return image
        ground = self.FuseMask(grounds)
        vpx, vpy = self.findVanPoint(ground)

        ground_depth_map = self.ground_depth(ground, (vpx, vpy))

        instance_depth_map_list = self.instance_depth(Instancemasks, (vpx, vpy), ground_depth_map)
        instance_depth_map = self.FuseMask(instance_depth_map_list)

        stuff_depth_map_list = self.stuff_depth(nongrounds, Instancemasks, (vpx, vpy), instance_depth_map)
        stuff_depth_map = self.FuseMask(stuff_depth_map_list)
        stuff_depth_map = self.Rearrage(stuff_depth_map)
        print("=" * 10)
        print("Detect Stuff : ",  len(all_stuffs))
        print("\t Stuff list")
        for info in Semanticinfos :
            print(self.stuff_list[info['category_id']])
        print("Detect Things : ",  len(all_instances))
        print("Ground Candidate : ", len(grounds))
        print("=" * 10)

        return ground_depth_map * ground + instance_depth_map + stuff_depth_map
    
    def ground_depth(self, ground, vp) :
        vpx, vpy = vp
        depth_map = np.zeros((self.height, self.width), dtype = np.float32)
        for y in range(self.height):
            if np.sum(ground[y,:]) != 0 :
                depth = self.dmax * (y - vpy) / vpy
                depth_map[y, :] = np.array([depth] * self.width)

        return depth_map
    
    def instance_depth(self, Instancemasks, vp, ground_depth_map):
        vpx, vpy = vp
        resoluation = self.dmax / vpy

        instance_depth_map = list()
        for mask in Instancemasks:
            mask = mask.astype(np.float32)
            mask_y, mask_x = np.nonzero(mask) # y, x
            depth_map = np.zeros((self.height, self.width), dtype = np.float32)

            bottom_y, bottom_x = mask_y[-1], mask_x[-1]
            depth = ground_depth_map.item(bottom_y, bottom_x)

            start_x = min(mask_x)
            end_x = max(mask_x)
        
            if bottom_y > vpy :
                if bottom_x < vpx:
                    for x in range(start_x, end_x + 1):
                        if x <= bottom_x :
                            depth = ground_depth_map[bottom_y, bottom_x]    
                        if x > bottom_x :
                            depth = ground_depth_map[bottom_y, bottom_x] - np.log(x - bottom_x) * resoluation
                        depth_map[:, x] = mask[:, x] * depth
                elif vpx <= bottom_x:
                    for x in range(start_x, end_x + 1):
                        if x < bottom_x :
                            depth = ground_depth_map[bottom_y, bottom_x] - np.log(bottom_x - x) * resoluation
                        if x >= bottom_x :
                            depth = ground_depth_map[bottom_y, bottom_x]  
                        depth_map[:, x] = mask[:, x] * depth
                instance_depth_map.append(depth_map)

        return instance_depth_map
                 
    def stuff_depth(self, nongrounds, Instancemasks, vp, instance_depth_map):
        vpx, vpy = vp

        X = list()
        Y = list()

        for idx, mask in enumerate(Instancemasks):
            cen = np.nonzero(mask)   # y, x
            ceny, cenx = np.average(cen, axis = 1) 
            
            dist = math.dist((vpx, vpy), (cenx, ceny))
            depth = instance_depth_map[int(ceny + 0.5), int(cenx + 0.5)]

            X.append(dist)
            Y.append(depth)
        
        dtype = torch.float
        X_tensor = torch.tensor(X, dtype=dtype, device= self.device)
        Y_tensor =  torch.tensor(Y, dtype=dtype, device= self.device)
        dataset = DummyData(X_tensor.reshape(-1, 1), Y_tensor.reshape(-1, 1))

        criterion = nn.SmoothL1Loss()
        depth_pred_model = PolynomialModel(degree=1).to(self.device)
        optimizer = Adam(depth_pred_model.parameters(), weight_decay=0.00001)
        for epoch in range(200):
            running_loss = train_step(model=depth_pred_model,
                                    data=dataset,
                                    optimizer=optimizer,
                                    criterion=criterion)
        
        stuff_depth_map = list()
        for mask in nongrounds:
            _, labels, stats, centroids \
            = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            labels = np.array(labels, dtype = np.float32)

            for idx, center in enumerate(centroids) :
                if idx == 0 : continue
                cen_x, cen_y = center
                dist = math.dist((cen_x, cen_y), (vpx, vpy))

                dist = torch.tensor([dist])
                pred = depth_pred_model(Variable(dist)).cpu().data.numpy()[0]

                Seg_mask = (labels == idx).astype(np.float32)
                
                mask_y, mask_x = np.nonzero(Seg_mask)
                start_x = min(mask_x)
                end_x = max(mask_x)
                Seg_mask[:, start_x] *= depth
                for x in range(start_x + 1, end_x + 1) :
                    Seg_mask[:, x] *= depth - np.log(abs(x - start_x)) * self.dmax / vpy
                
                stuff_depth_map.append(Seg_mask)

        return stuff_depth_map
    
    def Rearrage(self, depth_map):
        _max = depth_map.max()
        _min = depth_map.min()

        for y in range(self.height) :
            for x in range(self.width) :
                depth = depth_map.item(y, x)
                depth = ((depth -_min) / (_max - _min)) * self.dmax
                depth_map.itemset(y, x, depth)
        return depth_map
    
    def findVanPoint(self, imgMask) :
        vertical = np.sum(imgMask, axis = 1)
        vp_y = min(np.where(vertical != 0)[0])

        horzline = imgMask[vp_y,:]
        horzline = np.where(horzline != 0)[0]
        vp_x = int(np.average(horzline))

        return [vp_x, vp_y]

    # find Ground
    def findGround_Sky(self, masks, infos):
        grounds_list = ['gravel',
                        'platform',
                        'playingfield',
                        'railroad',
                        'road',
                        'sand',
                        'floor',
                        'pavement',
                        'grass',
                        'dirt',
                        'rug']
        ground_list = list()
        nonground_list = list()
        sky = None
        for idx, info in enumerate(infos) :
            if self.stuff_list[info['category_id']] in grounds_list :
                ground_list.append(masks[idx].astype(np.float32))
            elif self.stuff_list[info['category_id']] in 'sky' :
                sky = masks[idx].astype(np.float32)
            else :
                nonground_list.append(masks[idx].astype(np.float32))
        return ground_list, sky, nonground_list

    def FuseMask(self, masks):
        total = np.zeros((self.height, self.width), dtype = np.float32)

        for mask in masks :
            total += mask
        return total
