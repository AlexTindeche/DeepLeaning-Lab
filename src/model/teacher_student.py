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

from src.utils.rkd.loss import AttentionTransfer, RKdAngle, RkdDistance


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
            pretrained_weights_teacher="../checkpoints/empm-base.ckpt",
            pretrained_weights_student=None,
            batch_size=64,
        ) -> None:

        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.save_hyperparameters(ignore=["teacher_model", "student_model"])

        self.history_steps = historical_steps
        self.future_steps = future_steps
        self.submission_handler = SubmissionAv2()

        self.teacher = teacher_model
        self.student = student_model

        if pretrained_weights_teacher is not None:
            self.teacher.load_from_checkpoint(pretrained_weights_teacher)
        
        if pretrained_weights_student is not None:
            self.student.load_from_checkpoint(pretrained_weights_student)

        self.ce_weight = loss_weights["ce"]
        self.kd_weight = loss_weights["kd"]
        self.dist_criterion = RkdDistance()
        self.angle_criterion = RKdAngle()
        self.attention_criterion = AttentionTransfer()
        self.dist_ratio = loss_weights["rkd_distance"]
        self.angle_ratio = loss_weights["rkd_angle"]
        # NOT USED CURRENTLY
        self.attention_ratio = loss_weights["attention_ratio"]

        self.proj_layers = nn.ModuleList()
        self._projections_initialized = False
        
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
    
    def cal_loss(self, out, data, batch_idx=0):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        loss = 0
        B = y_hat.shape[0]
        B_range = range(B)


        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(-1)

        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[B_range, best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)

        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
        loss += agent_reg_loss + agent_cls_loss
        
        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.history_steps:]
        others_reg_loss = F.smooth_l1_loss(y_hat_others[others_reg_mask], y_others[others_reg_mask])
        loss += others_reg_loss
    
        return {
            "loss": loss,
            "reg_loss": agent_reg_loss.item(),
            "cls_loss": agent_cls_loss.item(),
            "others_reg_loss": others_reg_loss.item(),
        }


    def training_step(self, batch, batch_idx):
        # Teacher forward (no gradients)
        with torch.no_grad():
            hidden_embbeds_teacher, teacher_outputs = self.teacher(batch, get_embeddings=True)
        
        # Student forward
        hidden_embbeds_student, student_outputs = self.student(batch, get_embeddings=True)
        
        losses = self.cal_loss(student_outputs, batch, 
                               kd_loss = 0)
                               #kd_loss=self.rdk_loss(hidden_embbeds_student, hidden_embbeds_teacher))
        
        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return losses["loss"]
    
    def rdk_loss(self, student_embeds, teacher_embeds):
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
            for j in range(self.batch_size):
                student_emb = student_embeds[i][j]
                teacher_emb = teacher_embeds[i][j]

                # Apply projection if needed
                if self.proj_layers[i] is not None:
                    student_emb_projected = self.proj_layers[i](student_emb)


                dist_loss = self.dist_ratio * self.dist_criterion(student_emb_projected, teacher_emb)
                angle_loss = self.angle_ratio * self.angle_criterion(student_emb_projected, teacher_emb)
                loss += dist_loss + angle_loss

        loss = loss / self.batch_size
        return loss
    
    def _init_projection_layers(self, student_embeds, teacher_embeds):
        """
        Initialize projection layers if dimensions differ
        """
        self.proj_layers = nn.ModuleList()
        for s, t in zip(student_embeds, teacher_embeds):
            s_dim = s.shape[-1]
            t_dim = t.shape[-1]
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

    
    def cal_loss(self, out, data, batch_idx=0, kd_loss=0.):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        loss = 0
        B = y_hat.shape[0]
        B_range = range(B)


        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(-1)

        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[B_range, best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)

        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
        loss += agent_reg_loss + agent_cls_loss
        
        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.history_steps:]
        others_reg_loss = F.smooth_l1_loss(y_hat_others[others_reg_mask], y_others[others_reg_mask])
        loss += others_reg_loss

        loss += self.kd_weight * kd_loss
    
        return {
            "loss": loss,
            "reg_loss": agent_reg_loss.item(),
            "cls_loss": agent_cls_loss.item(),
            "rdk_loss": kd_loss,
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