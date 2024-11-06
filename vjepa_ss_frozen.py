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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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

def visualize_attention_with_pca_as_video(attn_maps, output_path, image_shape=(224, 224), token_grid=(32, 49)):
    """
    Visualizes high-dimensional attention maps with PCA and saves them as a video.

    Parameters:
        attn_maps (torch.Tensor): Attention maps of shape [1, num_frames, num_tokens, num_tokens]
        output_path (str): Path to save the output video
        image_shape (tuple): Final output image resolution for each frame (e.g., 224x224)
        token_grid (tuple): Shape of the token grid (e.g., 32x49)
    """
    num_frames = attn_maps.shape[1]
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5)
    
    pca = PCA(n_components=3)
    
    for frame_idx in range(num_frames):
        attn_map = attn_maps[0, frame_idx].cpu().numpy()
        
        attn_map_reduced = pca.fit_transform(attn_map.T).T  # Shape: [3, 1568]
        
        attn_rgb = [channel.reshape(token_grid) for channel in attn_map_reduced]
        attn_rgb = np.stack(attn_rgb, axis=-1)  # Shape: [32, 49, 3]
        attn_resized = attn_rgb
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min())
        attn_resized = (attn_resized * 255).astype(np.uint8)
        
        video_writer.write(attn_resized)

    video_writer.release()
    print(f"Video saved to {output_path}")
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
        use_sdpa=use_sdpa,)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    # TODO what clipaggregation do ? 
    data_path = "/home/vgl/emir/datasets/tudl/train_real/segmentation_video"
    obj_paths = [os.path.join(data_path, i) for i in sorted(os.listdir(data_path))]
    video_paths = [os.path.join(i, 'input.mp4') for i in obj_paths]
    mask_paths = [os.path.join(i, 'mask.mp4') for i in obj_paths]
    decord.bridge.set_bridge('torch')
    # input video loader contains 16 frames per clip and also we should batch them up to batch size
    from utils import VideoDataset, get_dataloaders
    # vd = VideoDataset(video_paths, mask_paths, frame_step, frames_per_clip)
    # train_loader, val_loader = get_dataloaders(vd, batch)
    segmentation_head = SegmentationHead3D(1024, 1).to(device)
    segmentation_head.load_state_dict(torch.load('/home/vgl/emir/weights/my_seg/segmentation_head_weights_2.pth'))
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(segmentation_head.parameters(), lr=1e-4)
    # for epoch in range(100):
    #     # with torch.no_grad():
    #     for x, y in train_loader:
    #         # 8, 3, 16, 224, 224
    #         vis_x = x.permute(0, 2, 3, 4, 1)
    #         cv2.imwrite('input.png', np.array(vis_x[0, 0]))
    #         x = x.to(device)
    #         y = y.to(device)
    #         encoder_output = encoder(x)
    #         features = encoder_output.view(8, 8, 14, 14, 1024)
    #         features = features.permute(0, 4, 1, 2, 3)
    #         print(features.shape) # 8x1024x8x14x14
    #         output = segmentation_head(features) # torch.Size([8, 16, 64, 224, 224])
    #         output = output.squeeze(1)
    #         print(output.shape)
    #         probs = torch.sigmoid(output)
    #         loss = criterion(output, y)
    #         binary_mask = (probs.detach().cpu().numpy() > 0.5)
    #         cv2.imwrite('output.png', binary_mask[0, 0]*255)
    #         print(f"Loss: {loss.item()}")
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    # torch.save(segmentation_head.state_dict(), '/home/vgl/emir/weights/my_seg/segmentation_head_weights_2.pth')
    # BELOW IS THE INFERENCE
    # pca_s = PCA(n_components=3)
    output_path = "/home/vgl/emir/segmentation_out_2_0000001.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, 30, (224, 224))

    vr = VideoReader("/home/vgl/emir/datasets/tudl/train_real/segmentation_video/000001/input.mp4", ctx=cpu(0), height=224, width=224)
    with torch.no_grad():
        for i in range(0, len(vr) - 16, 16):
            batch = vr.get_batch(range(i, i+16))
            batch = batch.to(device).unsqueeze(0).float()
            print(batch.permute(0, 4, 1, 2, 3).shape)
            out = encoder(batch.permute(0, 4, 1, 2, 3))
            # print(out.shape)
            features = out.view(1, 8, 14, 14, 1024)
            features = features.permute(0, 4, 1, 2, 3)
            output = segmentation_head(features)
            output = output.squeeze(1)
            print(features.shape) # 8x1024x8x14x14
            print(output.shape)
            probs = torch.sigmoid(output)
            binary_mask = (probs.detach().cpu().numpy() > 0.5)
        #     attention = attention.squeeze(0).detach().cpu().numpy()  # Shape: [16, 1568, 1568]
        #     for j in range(16):
        #         attn_j = attention[j]  # 1568x1568
                
        #         result = pca_s.fit_transform(attn_j.T).T
        #         attn_rgb = np.stack([channel.reshape(32, 49) for channel in result], axis=-1)
                
        #         attn_rgb_resized = cv2.resize(attn_rgb, (224, 224))
                
        #         attn_rgb_resized = (attn_rgb_resized - attn_rgb_resized.min()) / (attn_rgb_resized.max() - attn_rgb_resized.min())
        #         attn_rgb_resized = (attn_rgb_resized * 255).astype(np.uint8)
                
        #         output_video.write(attn_rgb_resized)
            
        # output_video.release()
                
            for j, frame in enumerate(batch.squeeze(0).cpu().numpy()):
                frame = frame.astype(np.uint8)
                mask = binary_mask[0, j]
                mask = (mask * 255).astype(np.uint8)
                
                colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                colored_mask[:, :, 0] = 0
                colored_mask[:, :, 2] = 0

                overlayed_frame = cv2.addWeighted(frame, 0.9, colored_mask, 0.3, 0)
                output_video.write(overlayed_frame)

        output_video.release()
            
        
    
        
        
        
        
        

        
        
if __name__ == '__main__':
    main()   