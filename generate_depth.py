import argparse
import multiprocessing as mp
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from demo.predictor import VisualizationDemo 
from projects.PanopticFCN.panopticfcn import add_panopticfcn_config
from projects.PanopticFCN.seg_depth.depth import DepthPredictor
from detectron2.utils.visualizer import _PanopticPrediction

def get_parser():
    parser = argparse.ArgumentParser(description="Segmentation and Depth map")
    parser.add_argument(
        "--config-file",
        default="projects/PanopticFCN/configs/PanopticFCN-Star-R50-3x-FAST.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="image root dir",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--WEIGHTS",
        default = "/content/drive/MyDrive/panoptic_fcn_star_r50_3x.pth",
        help="Pre-trained Model Weight",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser
def setup_cfg(args):
    cfg = get_cfg()
    add_panopticfcn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.WEIGHTS
    
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    depth_generator = DepthPredictor(cfg, args, logger, 255.)
    
    image_list = os.listdir(args.input[0])
    os.makedirs(args.output, exist_ok = True)
    for idx, image_name in enumerate(image_list) :
        image_path = os.path.join(args.input[0], image_name)
        image = read_image(image_path, format="BGR")

        predictions, visualized_output = demo.run_on_image(image)
        visualized_output.save(args.output + '/Seg_' + image_name)
        
        depth_map = depth_generator.predict(image)

        plt.figure()
        plt.imshow(depth_map)
        plt.axis('off')
        plt.colorbar()
        plt.savefig(args.output + '/depth_' + image_name)