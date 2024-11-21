from decord import VideoReader, cpu, VideoLoader
import numpy as np
import itertools
import decord
from torch.utils.data import DataLoader, Dataset
import json
import os
import open3d as o3d
import torch
import cv2

def get_dataloaders(dataset, batch_size):
    train_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=custom_collate_fn,
                    )
    val_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False)
    return train_loader, val_loader

def _test_video_loader(input):
    for batch in input:
        indices = np.array(batch[1][:, 0])
        if np.unique(indices).shape[0] != 1:
            print('VideoLoader returns overlapping video file indices')
            break
    print('VideoLoader does not return overlapping video file indices')
    
def custom_collate_fn(batch):
    x_list, bbox_data_list, depth_maps, label = zip(*batch)  # Unzip the batch
    x_batch = torch.stack(x_list, dim=0)  # Shape: [B, C, T, H, W]
    
    # Initialize lists to collect batched bbox data
    centers_batch = []
    dimensions_batch = []
    rotation_matrices_batch = []
    intrinsics_batch = []
    depth_maps = [torch.tensor(depth_map) for depth_map in depth_maps]
    for bbox_data in bbox_data_list:
        centers_batch.append(bbox_data['centers'])
        dimensions_batch.append(bbox_data['dimensions'])
        rotation_matrices_batch.append(bbox_data['rotation_matrices'])
        intrinsics_batch.append(bbox_data['intrinsics'])
    
    centers_batch = torch.stack(centers_batch, dim=0)            # Shape: [B, T, 3]
    dimensions_batch = torch.stack(dimensions_batch, dim=0)      # Shape: [B, T, 3]
    rotation_matrices_batch = torch.stack(rotation_matrices_batch, dim=0)  # Shape: [B, T, 3, 3]
    intrinsics_batch = torch.stack(intrinsics_batch, dim=0)      # Shape: [B, T, 3, 3]
    depth_maps = torch.stack(depth_maps, dim=0)
    bbox_data_batch = {
        'centers': centers_batch,
        'dimensions': dimensions_batch,
        'rotation_matrices': rotation_matrices_batch,
        'intrinsics': intrinsics_batch
    }
    
    return x_batch, bbox_data_batch, depth_maps, label


class Video3DBBoxDataset(Dataset):
    def __init__(self,
                 video_paths,
                 depth_paths:list,
                 frame_step,
                 K_jsons:list,
                 gt_jsons:list,
                 models:list,
                 frames_per_clip,
                 new_width=224,
                 new_height=224,
                 original_width=640,
                 original_height=480
    ):
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.new_width = new_width
        self.new_height = new_height
        self.original_width = original_width
        self.original_height = original_height
        self.depth_paths = depth_paths
        self.scale_w = self.new_width / self.original_width
        self.scale_h = self.new_height / self.original_height
        
        self._x = VideoLoader(
            video_paths,
            ctx=[cpu(0)],
            shape=(frames_per_clip, new_height, new_width, 3),
            interval=frame_step,
            skip=frame_step,
            shuffle=1
        )
        self._x = list(self._x)
        
        self.points_object = [
            np.asarray(o3d.io.read_triangle_mesh(model_path).vertices) for model_path in models
        ]
        
        self.intrinsics = []
        self.Rt = []
        for K_json in K_jsons:
            with open(K_json) as f:
                self.intrinsics.append(json.load(f))
        for gt_json in gt_jsons:
            with open(gt_json) as f:
                self.Rt.append(json.load(f))
    
    def __get_depth_maps__(self, f_n, q_ind):
        """
        loads depth maps from given folder,
        depth maps isn't provided by any video 
        to prevent and depth information loss
        """
        q_depth = self.depth_paths[f_n]
        q_depth_maps = sorted(os.listdir(q_depth))
        output_depth = []
        for frame_num in q_ind:
            depth_map = q_depth_maps[frame_num.item()]
            output_depth.append(
                cv2.resize(cv2.imread(os.path.join(q_depth, depth_map),cv2.IMREAD_UNCHANGED),
                           (self.new_width,
                            self.new_height)))
        return torch.tensor(np.array(output_depth))
    
    def __project_3d_points__(self, points_object, K, R, t):
        x_min_obj, y_min_obj, z_min_obj = points_object.min(axis=0)
        x_max_obj, y_max_obj, z_max_obj = points_object.max(axis=0)
        
        dimensions = np.array([
            x_max_obj - x_min_obj,
            y_max_obj - y_min_obj,
            z_max_obj - z_min_obj
        ])  # Shape: [3,]
        
        center_object = np.array([
            (x_min_obj + x_max_obj) / 2,
            (y_min_obj + y_max_obj) / 2,
            (z_min_obj + z_max_obj) / 2
        ])  # Shape: [3,]
        
        center_camera = R @ center_object + t.squeeze()
        
        rotation_matrix = R  # Shape: [3, 3]
        
        bbox_gt = {
            'center': center_camera,       # center of bbox in camera coordinates
            'dimensions': dimensions,      # 3 dimensionality of bbox [width, height, depth]
            'rotation_matrix': rotation_matrix,   # [3, 3]
            'K': K                         # Adjusted intrinsic matrix
        }
        return bbox_gt
    def __getitem__(self, idx):
        x, q_ind = self._x[idx]
        f_n = q_ind[0, 0].item()
        q_ind = q_ind[:, 1]
        
        centers = []
        dimensions = []
        rotation_matrices = []
        intrinsics = []
        
        for frame_num in q_ind:
            K_original = np.array(
                self.intrinsics[f_n][str(frame_num.item())]['cam_K']
            ).reshape(3, 3)
            
            K = K_original.copy()
            K[0, 0] *= self.scale_w  # fx
            K[0, 2] *= self.scale_w  # cx
            K[1, 1] *= self.scale_h  # fy
            K[1, 2] *= self.scale_h  # cy
            
            R = np.array(
                self.Rt[f_n][str(frame_num.item())][0]['cam_R_m2c']
            ).reshape(3, 3)
            t = np.array(
                self.Rt[f_n][str(frame_num.item())][0]['cam_t_m2c']
            ).reshape(3, 1)
            
            bbox_gt = self.__project_3d_points__(
                self.points_object[f_n], K, R, t
            )
            
            centers.append(bbox_gt['center'])
            dimensions.append(bbox_gt['dimensions'])
            rotation_matrices.append(bbox_gt['rotation_matrix'])
            intrinsics.append(bbox_gt['K'])
        
        centers = torch.tensor(np.array(centers))                 # Shape: [T, 3]
        dimensions = torch.tensor(np.array(dimensions))           # Shape: [T, 3]
        rotation_matrices = torch.tensor(np.array(rotation_matrices))  # Shape: [T, 3, 3]
        intrinsics = torch.tensor(np.array(intrinsics))           # Shape: [T, 3, 3]
    
        bbox_data = {
            'centers': centers,
            'dimensions': dimensions,
            'rotation_matrices': rotation_matrices,
            'intrinsics': intrinsics
        }
        output_depths = self.__get_depth_maps__(f_n, q_ind)
        return x.float(), bbox_data, output_depths.float(), f_n

    def __len__(self):
        return len(self._x)

class VideoDataset():
    def __init__(self,
                 video_paths,
                 mask_paths,
                 frame_step,
                 frames_per_clip
    ):
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        # TODO video transform like resize, and maybe view augmentation ?? etc.
        self._x = VideoLoader(video_paths, ctx=[cpu(0)], shape=(frames_per_clip, 224, 224, 3), interval=frame_step, skip=frame_step, shuffle=1)
        self._x = list(self._x)
        # NOTE do the same TODO for mask videos
        self._y = [VideoReader(mask_p, ctx=cpu(0), width=224, height=224) for mask_p in mask_paths]
    
    def __getitem__(self, idx):
        # decord.bridge.set_bridge('torch')
        x, q_ind =  self._x[idx]
        # print(q_ind)
        f_n = q_ind[0, 0].item()
        
        # print(f_n)
        q_ind = q_ind[:, 1]
        # print(q_ind)
        y = self._y[f_n].get_batch(q_ind)
        # x = 16, 640, 480, 3 -> C, T, H, W
        # remove channel dimension and make binary
        y = y.permute(3, 0, 1, 2)
        y = y[0]
        y = (y > 0).float()
        return x.permute(3, 0, 1, 2).float(), y.permute(0, 1, 2).float()
    def __len__(self):
        return len(self._x)

def _test_video_loader(input):
    for batch in input:
        indices = np.array(batch[1][:, 0])
        if np.unique(indices).shape[0] != 1:
            print('VideoLoader returns overlapping video file indices')
            break
    print('VideoLoader does not return overlapping video file indices')