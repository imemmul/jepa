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
from decord import VideoReader, cpu, VideoLoader
import decord
import cv2

resolution = 224
device = 'cuda'
pretrained_path= '/home/vgl/emir/weights/v-jepa_weights/vitl16.pth.tar'
model_name = 'vit_large'
patch_size = 16
tubelet_size = 2
frame_step = 4
frames_per_clip =  16
uniform_power = True
checkpoint_key = 'target_encoder'
use_SiLU = False
tight_SiLU = False
use_sdpa = False
attend_accross_segments = True
batch = 8

def main():
    depth_encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa,)
    decord.bridge.set_bridge('torch')
    vr = VideoReader(
        '/home/vgl/emir/datasets/tudl/train_real/segmentation_video/000001/depth_000001.mp4',
        ctx=cpu(0),
        width=resolution,
        height=resolution,)
    for i in range(0, len(vr)-frames_per_clip, frames_per_clip):
        batch = vr.get_batch(range(i, i+16))
        batch = batch.unsqueeze(0).unsqueeze(0).to(device)
        print(batch.shape)
        out = depth_encoder(batch)
    
    
    
        
        
        
        
        

        
        
if __name__ == '__main__':
    main()   