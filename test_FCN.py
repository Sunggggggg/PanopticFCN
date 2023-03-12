"""
현재 위치가 detectron2/
"""

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from projects.PanopticFCN.panopticfcn.gt_generate import GenerateGT
from projects.PanopticFCN.panopticfcn.panopticfcn.loss import sigmoid_focal_loss, weighted_dice_loss
from projects.PanopticFCN.panopticfcn.panopticfcn.head import build_position_head, build_kernel_head, build_feature_encoder, build_thing_generator, build_stuff_generator
from projects.PanopticFCN.panopticfcn.panopticfcn.backbone_utils import build_semanticfpn, build_backbone
from projects.PanopticFCN.panopticfcn.panopticfcn.utils import topk_score, multi_apply
__all__ = ["PanopticFCN"]

@META_ARCH_REGISTRY.register()
class PanopticFCN(nn.Module):
    """
    Implement PanopticFCN the paper :paper:`Fully Convolutional Networks for Panoptic Segmentation`.
    """
    def __init__(self, cfg):
        super().__init__()
        
        self.device                = torch.device(cfg.MODEL.DEVICE)
        # parameters
        self.cfg                   = cfg
        self.ignore_val            = cfg.MODEL.IGNORE_VALUE
        self.common_stride         = cfg.MODEL.SEMANTIC_FPN.COMMON_STRIDE

        self.center_top_num        = cfg.MODEL.POSITION_HEAD.THING.TOP_NUM
        self.weighted_num          = cfg.MODEL.POSITION_HEAD.THING.POS_NUM
        self.center_thres          = cfg.MODEL.POSITION_HEAD.THING.THRES
        self.sem_thres             = cfg.MODEL.POSITION_HEAD.STUFF.THRES
        self.sem_classes           = cfg.MODEL.POSITION_HEAD.STUFF.NUM_CLASSES
        self.sem_with_thing        = cfg.MODEL.POSITION_HEAD.STUFF.WITH_THING
        self.in_feature            = cfg.MODEL.FEATURE_ENCODER.IN_FEATURES
        self.inst_scale            = cfg.MODEL.KERNEL_HEAD.INSTANCE_SCALES

        self.pos_weight            = cfg.MODEL.LOSS_WEIGHT.POSITION
        self.seg_weight            = cfg.MODEL.LOSS_WEIGHT.SEGMENT
        self.focal_loss_alpha      = cfg.MODEL.LOSS_WEIGHT.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma      = cfg.MODEL.LOSS_WEIGHT.FOCAL_LOSS_GAMMA
        
        self.inst_thres            = cfg.MODEL.INFERENCE.INST_THRES
        self.panoptic_combine      = cfg.MODEL.INFERENCE.COMBINE.ENABLE
        self.panoptic_overlap_thrs = cfg.MODEL.INFERENCE.COMBINE.OVERLAP_THRESH
        self.panoptic_stuff_limit  = cfg.MODEL.INFERENCE.COMBINE.STUFF_AREA_LIMIT
        self.panoptic_inst_thrs    = cfg.MODEL.INFERENCE.COMBINE.INST_THRESH
        
        # backbone
        self.backbone              = build_backbone(cfg)
        self.semantic_fpn          = build_semanticfpn(cfg, self.backbone.output_shape())
        self.position_head         = build_position_head(cfg)
        self.kernel_head           = build_kernel_head(cfg)
        self.feature_encoder       = build_feature_encoder(cfg)
        self.thing_generator       = build_thing_generator(cfg)
        self.stuff_generator       = build_stuff_generator(cfg)
        self.get_ground_truth      = GenerateGT(cfg)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

        print(self.backbone)