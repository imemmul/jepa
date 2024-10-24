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
from decord import VideoReader, cpu

resolution = 224
device = 'cuda'
pretrained_path= '/home/vgl/emir/weights/v-jepa_weights/vitl16.pth.tar'
model_name = 'vit_large'
patch_size = 16
tubelet_size = 2
frames_per_clip = 16
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

    dataset = VideoDataset(
            input_video_path="/home/vgl/emir/datasets/tudl/train_real/segmentation_video",        
    )
    print(dataset[0][0].shape)

# TODO what i need is that instead of sampling linspace, rolling window should be more appropriate
class VideoDataset(torch.utils.data.Dataset):
    """ Video dataset for input video and corresponding mask video. """

    def __init__(
        self,
        input_video_path,
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        transform=None,
        mask_transform=None,
    ):
        self.video_paths = [os.path.join(input_video_path, i) for i in sorted(os.listdir(input_video_path))]
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.transform = transform
        self.mask_transform = mask_transform

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

    def __getitem__(self, index):
        video_path = os.path.join(self.video_paths[index], "input.mp4")
        mask_path = os.path.join(self.video_paths[index], "mask.mp4")
        video, video_indices = self.loadvideo_decord(video_path)
        mask, _ = self.loadvideo_decord(mask_path)  # Load the mask video

        # Apply transformations (if any)
        if self.transform is not None:
            video = self.transform(video)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return video, mask, video_indices

    def loadvideo_decord(self, video_path):
        """ Load video content using Decord """

        if not os.path.exists(video_path):
            warnings.warn(f'video path not found: {video_path}')
            return [], None

        _fsize = os.path.getsize(video_path)
        if _fsize < 1 * 1024:  # avoid hanging issue
            warnings.warn(f'video too short: {video_path}')
            return [], None

        try:
            vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None

        fpc = self.frames_per_clip
        fstp = self.frame_step
        clip_len = int(fpc * fstp)
        total_frames = len(vr)
        print(f"total frames: {total_frames}")
        # Ensure video has enough frames
        if len(vr) < clip_len:
            warnings.warn(f'video too short: {len(vr)} frames in {video_path}')
            return [], None

        vr.seek(0)
        clips = []
        clip_indices = []
        for i in range(0, total_frames - fpc + 1, fstp):
            indices = np.arange(i, i+fpc).astype(np.int64)
            buffer = vr.get_batch(indices).asnumpy()
            clips.append(buffer)
            clip_indices.append(indices)
        return clips, clip_indices
        # Below is used for sampling frames from video
        # indices = np.linspace(0, len(vr) - 1, num=fpc).astype(np.int64)
        # print(indices)
        # buffer = vr.get_batch(indices).asnumpy()

        # return buffer, indices
    def split_into_clips(self, video):
        """ Split video into clips of frames_per_clip size """
        clips = []
        num_frames = len(video)
        for i in range(0, num_frames - self.frames_per_clip + 1, self.frame_step):
            clip = video[i:i + self.frames_per_clip]
            clips.append(clip)
        return clips
    def __len__(self):
        return len(self.video_paths)


if __name__ == '__main__':
    main()   