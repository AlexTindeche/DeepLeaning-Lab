import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from pathlib import Path


from src.utils.optim import WarmupCosLR
from src.metrics import MR, brierMinFDE, minADE, minFDE
from src.utils.submission_av2 import SubmissionAv2

from src.utils.rkd.loss import AttentionTransfer, RKdAngle, RkdDistance, RKdAngleBatched, RKdAngleBatchedMemoryEff
from src.utils.rkd.rkd_utils import pdist_batched


torch.set_printoptions(sci_mode=False)


class TeacherStudentTrainer(pl.LightningModule):
    def __init__(
            self, 
            teacher_model,
            student_model, 
            loss_weights, 
            historical_steps=50,
            future_steps=60,
            lr=1e-3,
            weight_decay=1e-4,
            epochs=60,
            warmup_epochs=10,
            pretrained = True,
            pretrained_weights_teacher="./checkpoints/empm-base.ckpt",
            pretrained_weights_student=None,
            batch_size=64,
            student_embed_dim=[128, 128, 128, 128],
            teacher_embed_dim=[128, 128, 128, 128],
        ) -> None:

        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.save_hyperparameters()

        self.history_steps = historical_steps
        self.future_steps = future_steps
        self.submission_handler = SubmissionAv2()

        self.teacher = teacher_model
        self.student = student_model
        self.student_embed_dim = student_embed_dim
        self.teacher_embed_dim =teacher_embed_dim

        if pretrained_weights_teacher is not None:
            self.teacher.load_from_checkpoint(pretrained_weights_teacher)
        
        if pretrained_weights_student is not None:
            self.student.load_from_checkpoint(pretrained_weights_student)

        # self.ce_weight = loss_weights["ce"]
        # self.kd_weight = loss_weights["kd"]
        self.dist_criterion = RkdDistance()
        self.angle_criterion = RKdAngle()
        self.attention_criterion = AttentionTransfer()
        # self.dist_ratio = loss_weights["rkd_distance"]
        # self.angle_ratio = loss_weights["rkd_angle"]
        self.tau = loss_weights["tau"]
        self.NEG_RATIO = loss_weights["NEG_RATIO"]
        # NOT USED CURRENTLY
        self.attention_ratio = loss_weights["attention_ratio"]

        self._init_projection_layers()
        
        # Freeze teacher if pre-trained
        if pretrained:
            for param in self.teacher.parameters():
                    param.requires_grad = False
                
        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
                "brier-minFDE6": brierMinFDE(k=6)
            }
        )
        self.val_metrics = metrics.clone(prefix="val_")
        self.curr_ep = 0
        return
    

    def getNet(self):
        return self.student

    def forward(self, data):
        return self.student(data)
    
    def predict(self, data, full=False, teacher=False):
        with torch.no_grad():
            if teacher:
                out = self.teacher(data)
            else:
                out = self.student(data)
        predictions, prob = self.submission_handler.format_data(
            data, out["y_hat"], out["pi"], inference=True
        )
        predictions = [predictions, out] if full else predictions
        return predictions, prob   


    def training_step(self, batch, batch_idx):
        
        # Teacher forward (no gradients)
        with torch.no_grad():
            hidden_embbeds_teacher, teacher_outputs = self.teacher(batch, get_embeddings=True)
        
        # Student forward
        hidden_embbeds_student, student_outputs = self.student(batch, get_embeddings=True)
        
        losses = self.cal_loss(student_outputs, batch, teacher_out=teacher_outputs,
                               kd_loss=0)

        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            
        if batch_idx % 30 == 0:
            print(
                f"Epoch {self.curr_ep} | Batch {batch_idx} | "
                f"Critic pi loss: {losses['critic_pi_loss']:.4f} | "
                f"Critic loc loss: {losses['critic_loc_loss']:.4f}"
            )

        return losses["loss"]
    
    def rdk_loss(self, student_embeds, teacher_embeds, sample_k=2):
        """ 
        Calculate the RKD loss between student and teacher embeddings.
        """
        loss = 0

        # Check if projection layers are initialized
        # If not, initialize them based on the dimensions of student and teacher embeddings
        # This is done lazily to assuming that the dimensions of embeddings will not change during training, but are unknown at initialization.
        # This allows for flexibility in the model architecture.
        if not self._projections_initialized:
            self._init_projection_layers(student_embeds, teacher_embeds)

        for i in range(len(student_embeds)):
            student_emb = student_embeds[i]
            teacher_emb = teacher_embeds[i]

            # Apply projection if needed
            if self.proj_layers[i] is not None:
                student_emb_projected = self.proj_layers[i](student_emb)

             # Method 1: Random mask
            mask = self.create_random_mask(student_emb.shape[0], sample_k, student_emb.device)
            
            # Apply mask to get only the selected embeddings
            masked_student = student_emb_projected[mask].view(-1, student_emb_projected.shape[-1])
            masked_teacher = teacher_emb[mask].view(-1, teacher_emb.shape[-1])
            # student_flat = masked_student.reshape(-1, teacher_emb.shape[-1])
            # teacher_flat = masked_teacher.reshape(-1, teacher_emb.shape[-1])
            dist_loss = self.dist_ratio * self.dist_criterion(masked_student, masked_teacher)
            angle_loss = self.angle_ratio * self.angle_criterion(masked_student, masked_teacher)
            loss = loss + dist_loss + angle_loss

        return loss
    
    def h_score(self, t_feat, s_feat, tau=0.1, N_over_M=1.0):
        """
        Compute h(T, S) as defined in Eq (19) in the paper.
        """
        # Normalize features 
        t_feat = F.normalize(t_feat, dim=1)
        s_feat = F.normalize(s_feat, dim=1)

        # Similarity scores
        sim = torch.sum(t_feat * s_feat, dim=1) / tau  # Exponent of e for numerator
        numerator = torch.exp(sim)
        denominator = numerator + N_over_M
        h = numerator / denominator  # shape: (B,)
        return h    
    
    def critic_loss(self, t_pos, s_pos, t_neg, s_neg, tau=0.1, N=1.0, M = 1.0):
        """
        Compute the full critic loss:
        - t_pos, s_pos: matched teacher/student features (C=1)
        - t_neg, s_neg: mismatched features (C=0)
        """
        N_over_M = N / M if M != 0 else N / (M + 1e-8)  # Avoid division by zero
        
        # Positive pair scores
        h_pos = self.h_score(t_pos, s_pos, tau, N_over_M)
        loss_pos = torch.median(torch.log(h_pos + 1e-8))  # Explicit Monte Carlo estimate

        # Negative pair scores
        h_neg = self.h_score(t_neg, s_neg, tau, N_over_M)
        loss_neg = torch.median(torch.log(1 - h_neg + 1e-8))  # Explicit Monte Carlo estimate

        # Combine as in Eq (18)
        loss = -(loss_pos + N * loss_neg)
        return loss
    
    
    def contrastive_loss_batch_online(self, teacher_feats, student_feats, tau=0.1, NEG_RATIO=0.1):
        """
        Online contrastive loss function using custom critic loss.
        - Negative pairs: randomly sample neg_size pairs (i, j) where i != j
        - Positive pairs: use pos_size pairs (i, i)
        """
        batch_size = student_feats.shape[0]
        device = student_feats.device

        if batch_size == 1:
            return F.mse_loss(student_feats, teacher_feats)

        neg_size = int(batch_size * NEG_RATIO)
        pos_size = batch_size - neg_size

        # --- Positive pairs (i == j) ---
        pos_indices = torch.arange(pos_size, device=device)
        t_pos = teacher_feats[pos_indices]
        s_pos = student_feats[pos_indices]

        # --- Negative pairs (i != j) ---
        neg_indices_1 = torch.randint(0, batch_size, (neg_size,), device=device)
        neg_indices_2 = torch.randint(0, batch_size, (neg_size,), device=device)
        # Ensure i != j for negative pairs
        mask = neg_indices_1 != neg_indices_2
        while not torch.all(mask):
            num_to_replace = int((~mask).sum().item())
            neg_indices_2[~mask] = torch.randint(0, batch_size, (num_to_replace,), device=device)
            mask = neg_indices_1 != neg_indices_2

        t_neg = teacher_feats[neg_indices_1]
        s_neg = student_feats[neg_indices_2]

        # Compute critic loss
        loss = self.critic_loss(
            t_pos, s_pos, t_neg, s_neg, tau=tau, N=neg_size, M=pos_size
        )
        return loss


    
    def _init_projection_layers(self):
        """
        Initialize projection layers if dimensions differ
        """
        self.proj_layers = nn.ModuleList()
        for s, t in zip(self.student_embed_dim, self.teacher_embed_dim):
            s_dim = s
            t_dim = t
            if s_dim != t_dim:
                # Small MLP: 2-layer projection (s_dim → t_dim)
                self.proj_layers.append(
                    nn.Sequential(
                        nn.Linear(s_dim, t_dim),
                        nn.ReLU(),
                        nn.Linear(t_dim, t_dim)
                    ).to(self.device)  # Ensure the layer is on the correct device
                )
            else:
                self.proj_layers.append(None)  # No need to project

        self._projections_initialized = True

    def create_random_mask(self, N, sample_k, device):
        """
        Create a random boolean mask that selects sample_k out of N elements.
        
        Args:
            N: total number of elements
            sample_k: number of elements to keep (True)
            device: device for the mask tensor
        
        Returns:
            mask: [N] boolean tensor with sample_k True values
        """
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        indices = torch.randperm(N, device=device)[:sample_k]
        mask[indices] = True
        return mask
    
    def cal_loss(self, out, data, batch_idx=0, kd_loss=0., teacher_out=None):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        teacher_loc_emb = None
        teacher_pi_emb = None
        # Extract teacher outputs
        if teacher_out is not None:
            teacher_loc_emb = teacher_out["loc_emb"].detach() if "y_hat" in teacher_out else None
            teacher_pi_emb = teacher_out["pi_emb"].detach() if "y_hat" in teacher_out else None

        loss = 0
        B = y_hat.shape[0]
        B_range = range(B)


        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(-1)

        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[B_range, best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)

        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
        loss = loss + agent_reg_loss + agent_cls_loss
        
        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.history_steps:]
        others_reg_loss = F.smooth_l1_loss(y_hat_others[others_reg_mask], y_others[others_reg_mask])
        loss = loss + others_reg_loss

        if teacher_loc_emb is not None:
            # --- Knowledge Distillation Loss on location prediction embeddings---
            
            # if self.teacher.embed_dim != self.student.embed_dim:
            #     # Apply projection layers if dimensions differ
            #     student_loc_emb = self.proj_layers[-1](out["loc_emb"])
            # else:
            student_loc_emb = out["loc_emb"]
            critic_loss = 0
            # print(teacher_loc_emb.shape, student_loc_emb.shape)
            for actor in range(teacher_loc_emb.shape[1]):
                # rkd_loss += RKdAngle()(teacher_loc_emb[:, actor], student_loc_emb[:, actor])
                # rkd_loss_distance += RkdDistance()(teacher_loc_emb[:, actor], student_loc_emb[:, actor])
                critic_loss += self.contrastive_loss_batch_online(
                    teacher_loc_emb[:, actor], student_loc_emb[:, actor],
                    tau=self.tau,
                    NEG_RATIO=0.2
                )
            # print("critic_loss loc", critic_loss / teacher_loc_emb.shape[1])
            critic_loc_loss = critic_loss / teacher_loc_emb.shape[1]
            loss = loss + critic_loc_loss

        if teacher_pi_emb is not None:
            # --- Knowledge Distillation Loss on pi embeddings---
            # if self.teacher.embed_dim != self.student.embed_dim:
            #     # Apply projection layers if dimensions differ
            #     student_loc_emb = self.proj_layers[-1](out["loc_emb"])
            # else:
            student_loc_emb = out["loc_emb"]
            critic_loss = 0
            for actor in range(teacher_loc_emb.shape[1]):
                critic_loss += self.contrastive_loss_batch_online(
                    teacher_loc_emb[:, actor], student_loc_emb[:, actor],
                    tau=self.tau,
                    NEG_RATIO=self.NEG_RATIO
                )
            # print("critic_loss pi", critic_loss / teacher_loc_emb.shape[1])
            critic_pi_loss = critic_loss / teacher_loc_emb.shape[1]
            loss = loss + critic_pi_loss

    
        return {
            "loss": loss,
            "reg_loss": agent_reg_loss.item(),
            "cls_loss": agent_cls_loss.item(),
            "critic_pi_loss": critic_pi_loss.item() if teacher_pi_emb is not None else 0,
            "critic_loc_loss": critic_loc_loss.item() if teacher_loc_emb is not None else 0,
            "others_reg_loss": others_reg_loss.item(),
        }
    
    def validation_step(self, data, batch_idx):
        out = self(data)

        losses = self.cal_loss(out, data, -1)
        metrics = self.val_metrics(out, data["y"][:, 0])

        self.log(
            "val/reg_loss",
            losses["reg_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        for k in self.val_scores.keys(): self.val_scores[k].append(metrics[k].item())

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

    def on_train_start(self) -> None:
        torch.autograd.set_detect_anomaly(True)

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)

    def test_step(self, data, batch_idx) -> None:
        out = self(data)
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def on_validation_start(self) -> None:
        self.val_scores = {"val_MR": [], "val_minADE1": [], "val_minADE6": [], "val_minFDE1": [], "val_minFDE6": [], "val_brier-minFDE6": []}

    def on_validation_end(self) -> None:      
        print( " & ".join( ["{:5.3f}".format(np.mean(self.val_scores[k])) for k in ["val_MR", "val_minADE6", "val_minFDE6", "val_brier-minFDE6"]] ) )
        self.curr_ep += 1

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
            nn.GRUCell
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
            nn.Parameter
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]