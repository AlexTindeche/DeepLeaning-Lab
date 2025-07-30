import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.rkd.rkd_utils import pdist


__all__ = ['AttentionTransfer', 'RkdDistance', 'RKdAngle']

class AttentionTransfer(nn.Module):
    def forward(self, student, teacher):
        s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))

        with torch.no_grad():
            t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.size(0), -1))

        return (s_attention - t_attention).pow(2).mean()


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss
    
# More memory-efficient version that processes in chunks
class RKdAngleBatchedMemoryEff(nn.Module):
    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient version that avoids creating the full N^4 tensor.
        """
        B, N, D = student.shape
        
        # Compute pairwise differences: [B, N, N, D]
        with torch.no_grad():
            td = teacher.unsqueeze(2) - teacher.unsqueeze(1)  # [B, N, N, D]
            norm_td = F.normalize(td, p=2, dim=-1)            # [B, N, N, D]

        sd = student.unsqueeze(2) - student.unsqueeze(1)      # [B, N, N, D]
        norm_sd = F.normalize(sd, p=2, dim=-1)                # [B, N, N, D]

        total_loss = 0.0
        count = 0
        
        # Process each batch sample separately to save memory
        for b in range(B):
            # Get normalized difference vectors for this batch: [N*N, D]
            td_b = norm_td[b].view(N * N, D)
            sd_b = norm_sd[b].view(N * N, D)
            
            # Compute angle matrix: [N*N, N*N]
            with torch.no_grad():
                t_angles_b = torch.mm(td_b, td_b.t()).view(-1)
            
            s_angles_b = torch.mm(sd_b, sd_b.t()).view(-1)
            
            # Accumulate loss
            total_loss += F.smooth_l1_loss(s_angles_b, t_angles_b, reduction='sum')
            count += s_angles_b.numel()
        
        return total_loss / count


class RKdAngleBatched(nn.Module):
    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """
        Batched RKD-Angle loss.
        student, teacher: [B, N, D]
        """
        B, N, D = student.shape

        # Compute pairwise differences: [B, N, N, D]
        with torch.no_grad():
            td = teacher.unsqueeze(2) - teacher.unsqueeze(1)  # [B, N, N, D]
            td_norm = F.normalize(td, p=2, dim=-1)            # [B, N, N, D]
            t_angle = torch.matmul(td_norm, td_norm.transpose(-1, -2))  # [B, N, N]

        sd = student.unsqueeze(2) - student.unsqueeze(1)      # [B, N, N, D]
        sd_norm = F.normalize(sd, p=2, dim=-1)                # [B, N, N, D]
        s_angle = torch.matmul(sd_norm, sd_norm.transpose(-1, -2))      # [B, N, N]

        # Flatten [B, N, N] → [B, N*N], then compute mean loss across batch
        loss = F.smooth_l1_loss(s_angle.view(B, -1), t_angle.view(B, -1), reduction='mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher, dist_func=pdist):
        with torch.no_grad():
            t_d = dist_func(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = dist_func(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss