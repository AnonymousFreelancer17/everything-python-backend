import torch
import torch.nn as nn
from .generator import ImageGenerator

class MoCoGANGenerator(nn.Module):
    def __init__(self, z_dim=100, video_len=16):
        super().__init__()
        self.video_len = video_len
        self.content_dim = z_dim // 2
        self.motion_dim = z_dim // 2
        self.frame_gen = ImageGenerator(z_dim=z_dim)

    def forward(self, z):
        # z = [B, z_dim]
        B = z.size(0)
        content_code = z[:, :self.content_dim]  # [B, z_dim/2]
        motion_code = z[:, self.content_dim:]   # [B, z_dim/2]

        frames = []
        for _ in range(self.video_len):
            frame_z = torch.cat([content_code, motion_code], dim=1)
            frame = self.frame_gen(frame_z)
            frames.append(frame)

        return torch.stack(frames, dim=1)  # [B, T, C, H, W]
