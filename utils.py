from decord import VideoReader, cpu, VideoLoader
import numpy as np
import itertools
import decord
from torch.utils.data import DataLoader, Dataset

def get_dataloaders(dataset, batch_size):
    train_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True)
    val_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False)
    return train_loader, val_loader

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