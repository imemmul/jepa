# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# this implementation stands for semantic segmentation for v-jepa with frozen encoder
import torch
from evals.video_classification_frozen.eval import init_model
from src.models.segmenter import Segmenter
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
frame_step = 8
frames_per_clip =  16
uniform_power = True
checkpoint_key = 'target_encoder'
use_SiLU = False
tight_SiLU = False
use_sdpa = False
attend_accross_segments = True


        
def main():
    encoder = init_model(
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
        use_sdpa=use_sdpa)
    encoder = ClipAggregation(
        encoder,
        attend_across_segments=attend_accross_segments,).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    data_path = "/home/vgl/emir/datasets/tudl/train_real/segmentation_video"
    obj_paths = [os.path.join(data_path, i) for i in sorted(os.listdir(data_path))]
    video_paths = [os.path.join(i, 'input.mp4') for i in obj_paths]
    mask_paths = [os.path.join(i, 'mask.mp4') for i in obj_paths]
    decord.bridge.set_bridge('torch')
    input_vl = VideoLoader(video_paths, ctx=[cpu(0)], shape=(frames_per_clip, 640, 480, 3), interval=frame_step, skip=frame_step, shuffle=1)
    masks = [VideoReader(mask_p, ctx=cpu(0)) for mask_p in mask_paths]
    # TODO check if VideoLoader doesn't return any overlapping video file indices  like [0, 0, 0, 0, 0, 0, 1] etc. if it does, then we need to fix it
    for batch in input_vl:
        indices = batch[1]
        # print(indices)
        f_n = indices[0, 0].item()
        q_i = indices[:, 1]
        input_frames, mask_frames = batch[0].to(device), masks[f_n].get_batch(q_i).to(device)
        print(input_frames.shape, mask_frames.shape)
        outputs = encoder(input_frames.unsqueeze(0).unsqueeze(0))
        
        
if __name__ == '__main__':
    main()   