# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# this implementation stands for semantic segmentation for v-jepa with frozen encoder now its 3d object detection
import torch
from evals.video_classification_frozen.eval import init_model
import numpy as np
import warnings
import os
from utils_data import (
    Video3DBBoxDataset,
    get_dataloaders,)
import decord
import cv2
import sys
import sys
import os
import yaml
import torch.nn.functional as F
from loss import compute_loss, rotmat_to_quat
sys.path.append("/home/vgl/emir/vp-6d")
from models.multimodel_fusion import (CrossAttentionDecoder, 
                                      CrossAttentionDecoderLayer,
                                      generate_positional_embeddings,
                                      ObjectQueryDecoder,
                                      PredictionHead)
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

def compute_loss_trivial(predictions, ground_truth):
    """
    Compute the loss between predictions and ground truth.

    Args:
        predictions: Predicted bounding boxes, shape [batch_size, num_queries, num_outputs]
        ground_truth: Ground truth bounding boxes, shape [batch_size, num_objects, num_outputs]

    Returns:
        loss: Scalar tensor representing the loss
    """
    # For simplicity, we'll use L1 loss between predictions and ground truth after matching
    # Implement Hungarian matching here (or another matching algorithm)
    predicted_params = predictions[:, 0, :]  # [batch_size, num_outputs]
    gt_params = ground_truth[:, 0, :]        # [batch_size, num_outputs]

    # Compute L1 loss
    loss = F.l1_loss(predicted_params, gt_params)
    return loss

import numpy as np

def project_3d_bbox_to_2d(center, dimensions, rotation_matrix, K):
    """
    Projects a 3D bounding box onto the 2D image plane.

    Args:
        center: Center of the bounding box in camera coordinates, shape [3,]
        dimensions: Dimensions of the bounding box (width, height, depth), shape [3,]
        rotation_matrix: Rotation matrix of the bounding box, shape [3, 3]
        K: Camera intrinsics matrix, shape [3, 3]

    Returns:
        corners_2d: Array of projected 2D corners, shape [8, 2]
    """
    W, H, L = dimensions

    x_corners = W / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_corners = H / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = L / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])


    corners_object = np.vstack((x_corners, y_corners, z_corners))  # Shape: [3, 8]

    corners_camera = rotation_matrix @ corners_object  # Shape: [3, 8]

    corners_camera = corners_camera + center.reshape(3, 1)  # Shape: [3, 8]

    corners_homogeneous = K @ corners_camera  # Shape: [3, 8]
    corners_2d = corners_homogeneous[:2, :] / corners_homogeneous[2, :]  # Normalize by depth

    corners_2d = corners_2d.T  # Shape: [8, 2]

    return corners_2d

def draw_2d_bbox(image, corners_2d, color=(0, 255, 0)):
    """
    Draws a 3D bounding box projected onto a 2D image.

    Args:
        image: The image array.
        corners_2d: Array of projected 2D corners, shape [8, 2].
        color: Tuple representing the color of the bounding box lines.
    """
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    corners_2d = corners_2d.astype(int)

    for edge in edges:
        pt1 = tuple(corners_2d[edge[0]])
        pt2 = tuple(corners_2d[edge[1]])
        cv2.line(image, pt1, pt2, color=color, thickness=2)


from scipy.spatial.transform import Rotation

def visualize_predictions_with_ground_truth(x, predictions, bbox_data, sample_idx, save_dir):

    image = np.array(x)

    predicted_params = predictions[sample_idx].cpu().detach().numpy()
    gt_centers = bbox_data['centers'][sample_idx][sample_idx].cpu().numpy()
    gt_dimensions = bbox_data['dimensions'][sample_idx][sample_idx].cpu().numpy()
    gt_rotations = bbox_data['rotation_matrices'][sample_idx][sample_idx].cpu().numpy()
    K = bbox_data['intrinsics'][sample_idx][sample_idx].cpu().numpy()

    pred_params = predicted_params[sample_idx]
    pred_center = pred_params[:3]  # [X, Y, Z]
    pred_dimensions = pred_params[3:6]  # [W, H, L]
    pred_angles = pred_params[6:]  # quaternion [W, X, Y, Z]
    r = Rotation.from_quat(pred_angles)
    pred_rotation_matrix = r.as_matrix()  # Shape: [3, 3]
    pred_corners_2d = project_3d_bbox_to_2d(pred_center, pred_dimensions, pred_rotation_matrix, K)
    draw_2d_bbox(image, pred_corners_2d, color=(0, 0, 255))

    gt_corners_2d = project_3d_bbox_to_2d(gt_centers, gt_dimensions, gt_rotations, K)
    draw_2d_bbox(image, gt_corners_2d, color=(0, 255, 0))

    output_path = os.path.join(save_dir, 'pred_2.png')
    cv2.imwrite(output_path, image)


def main():
    depth_encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        in_channels=1,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa,)
    
    rgb_encoder = init_model(
        crop_size=resolution,
        device=device,
        in_channels=3,
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
    rgb_encoder.eval()
    for p in rgb_encoder.parameters():
        p.requires_grad = False
        
    video_paths = [
    "/home/vgl/emir/datasets/tudl/train_real/compressed_objects/segmentation_videos/000001/input.mp4",
    "/home/vgl/emir/datasets/tudl/train_real/compressed_objects/segmentation_videos/000002/input.mp4",
    "/home/vgl/emir/datasets/tudl/train_real/compressed_objects/segmentation_videos/000003/input.mp4",
    ]
    K_jsons = [
        "/home/vgl/emir/datasets/tudl/train_real/000001/scene_camera.json",
        "/home/vgl/emir/datasets/tudl/train_real/000002/scene_camera.json",
        "/home/vgl/emir/datasets/tudl/train_real/000003/scene_camera.json"
    ]
    gt_jsons = [
        "/home/vgl/emir/datasets/tudl/train_real/000001/scene_gt.json",
        "/home/vgl/emir/datasets/tudl/train_real/000002/scene_gt.json",
        "/home/vgl/emir/datasets/tudl/train_real/000003/scene_gt.json"
    ]
    models = [
        "/home/vgl/emir/datasets/tudl/tudl_models/models/obj_000001.ply",
        "/home/vgl/emir/datasets/tudl/tudl_models/models/obj_000002.ply",
        "/home/vgl/emir/datasets/tudl/tudl_models/models/obj_000003.ply"
    ]
    depth_paths = [
        "/home/vgl/emir/datasets/tudl/train_real/000001/depth",
        "/home/vgl/emir/datasets/tudl/train_real/000002/depth",
        "/home/vgl/emir/datasets/tudl/train_real/000003/depth"
    ]
    decord.bridge.set_bridge('torch')
    vd = Video3DBBoxDataset(
        video_paths=video_paths,
        frame_step=frame_step,
        K_jsons=K_jsons,
        depth_paths=depth_paths,
        gt_jsons=gt_jsons,
        models=models,
        frames_per_clip=frames_per_clip,
    )
    layer = CrossAttentionDecoderLayer(d_model=1024).to(device)
    decoder = CrossAttentionDecoder(num_layers=6, decoder_layer=layer).to(device)
    object_decoder = ObjectQueryDecoder(d_model=1024, num_queries=1, num_heads=8, num_layers=6).to(device)
    ph = PredictionHead(d_model=1024, num_outputs=10, frames_per_clip=frames_per_clip, num_queries=1).to(device)
    # print(decoder)
    optimizer = torch.optim.AdamW([
        {'params': depth_encoder.parameters()},
        {'params': decoder.parameters()},
        {'params': object_decoder.parameters()},
        {'params': ph.parameters()},
    ], lr=1e-4)
    train_loader, val_loader = get_dataloaders(vd, 8)
    for epoch in range(20):
        for batch in train_loader:
            x, bbox, depth, obj_id = batch # b, t, h, w, c -> b, c, t, h, w
            # print(f"obj: {obj_id}")
            save_rgb = x[0, 0]
            x, depth = x.to(device), depth.unsqueeze(1).to(device)
            # print(depth.shape)
            # print(x.shape)
            out_rgb = rgb_encoder(x.permute(0, 4, 1, 2, 3))
            batch_size, seq_len, d_model = out_rgb.shape
            out_depth = depth_encoder(depth)
            _, seq_len_depth, d_model_depth = out_depth.shape
            query_pos = generate_positional_embeddings(seq_len, d_model, device)
            query_pos = query_pos.unsqueeze(1).repeat(1, batch_size, 1)
            mem_pos = generate_positional_embeddings(seq_len_depth, d_model_depth, device)
            mem_pos = mem_pos.unsqueeze(1).repeat(1, batch_size, 1)
            tgt = out_rgb.transpose(0, 1)
            mem = out_depth.transpose(0, 1)
            mem_final = decoder(tgt, mem, pos=mem_pos, query_pos=query_pos)
            obj_feat = object_decoder(mem_final)
            predictions = ph(obj_feat)
            predictions = predictions.squeeze()
            print(f"predictions: {predictions.shape}")
            centers = bbox['centers'].to(device)
            dimensions = bbox['dimensions'].to(device)
            rotation_matrices = bbox['rotation_matrices'].to(device)
            quats = rotmat_to_quat(rotation_matrices)
            gt = torch.cat([centers, dimensions, quats], dim=-1) # 8x16x10
            print(f"gt: {gt.shape}")
            loss = compute_loss(predictions, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"loss: {loss.item()}")
            visualize_predictions_with_ground_truth(save_rgb, predictions, bbox, 0, ".")
    
        
        
        

        
        
if __name__ == '__main__':
    main()   