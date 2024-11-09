# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# this implementation stands for semantic segmentation for v-jepa with frozen encoder
import torch
from evals.video_classification_frozen.eval import init_model
from src.models.segmenter import SegmentationHead3D
from evals.video_classification_frozen.utils import (
    make_transforms,
    ClipAggregation,
    FrameAggregation
)
import numpy as np
import warnings
import os
# from utils import Video3DBBoxDataset, get_dataloaders
import decord
import cv2
import sys
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'MonoDETR')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), 'MonoDETR')))

from lib.helpers.model_helper import build_model
# resolution = 224
# device = 'cuda'
# pretrained_path= '/home/vgl/emir/weights/v-jepa_weights/vitl16.pth.tar'
# model_name = 'vit_large'
# patch_size = 16
# tubelet_size = 2
# frame_step = 4
# frames_per_clip =  16
# uniform_power = True
# checkpoint_key = 'target_encoder'
# use_SiLU = False
# tight_SiLU = False
# use_sdpa = False
# attend_accross_segments = True
# batch = 8


# def main():
#     depth_encoder = init_model(
#         crop_size=resolution,
#         device=device,
#         pretrained=pretrained_path,
#         model_name=model_name,
#         patch_size=patch_size,
#         tubelet_size=tubelet_size,
#         frames_per_clip=frames_per_clip,
#         uniform_power=uniform_power,
#         checkpoint_key=checkpoint_key,
#         use_SiLU=use_SiLU,
#         tight_SiLU=tight_SiLU,
#         use_sdpa=use_sdpa,)
#     video_paths = [
#     "/home/vgl/emir/datasets/tudl/train_real/compressed_objects/segmentation_videos/000001/input.mp4",
#     "/home/vgl/emir/datasets/tudl/train_real/compressed_objects/segmentation_videos/000002/input.mp4",
#     "/home/vgl/emir/datasets/tudl/train_real/compressed_objects/segmentation_videos/000003/input.mp4",
#     ]
#     K_jsons = [
#         "/home/vgl/emir/datasets/tudl/train_real/000001/scene_camera.json",
#         "/home/vgl/emir/datasets/tudl/train_real/000002/scene_camera.json",
#         "/home/vgl/emir/datasets/tudl/train_real/000003/scene_camera.json"
#     ]
#     gt_jsons = [
#         "/home/vgl/emir/datasets/tudl/train_real/000001/scene_gt.json",
#         "/home/vgl/emir/datasets/tudl/train_real/000002/scene_gt.json",
#         "/home/vgl/emir/datasets/tudl/train_real/000003/scene_gt.json"
#     ]
#     models = [
#         "/home/vgl/emir/datasets/tudl/tudl_models/models/obj_000001.ply",
#         "/home/vgl/emir/datasets/tudl/tudl_models/models/obj_000002.ply",
#         "/home/vgl/emir/datasets/tudl/tudl_models/models/obj_000003.ply"
#     ]
#     decord.bridge.set_bridge('torch')
#     # vd = Video3DBBoxDataset(
#     #     video_paths=video_paths,
#     #     frame_step=frame_step,
#     #     K_jsons=K_jsons,
#     #     gt_jsons=gt_jsons,
#     #     models=models,
#     #     frames_per_clip=frames_per_clip,
#     # )
#     # print(bbox[0]['rotation_matrix'].shape)
#     build_depthaware_transformer()
#     # train_loader, val_loader = get_dataloaders(vd, 8)
#     # for batch in train_loader:
#     #     x, bbox = batch
#     #     print(bbox['rotation_matrices'].shape)
#     #     print(bbox['intrinsics'].shape)
#     #     print(bbox['dimensions'].shape)
#     #     print(bbox['centers'].shape)
#     #     break
    
    
    
        
        
        
        
        

        
        
# if __name__ == '__main__':
#     main()   