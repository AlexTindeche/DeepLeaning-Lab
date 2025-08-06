import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from .emp import EMP
from src.metrics import MR, brierMinFDE, minADE, minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2 import SubmissionAv2


torch.set_printoptions(sci_mode=False)

class hintBasedTrainer(pl.LightningModule):
    def __init__(
        self,
        teacher_model=None,
        student_model=None, 
        historical_steps=50,
        future_steps=60,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        pretrained=True,
        pretrained_weights_teacher="checkpoints/empm.ckpt",
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

        self.student = student_model
        self.teacher = teacher_model

        if pretrained_weights_teacher is not None:
            self.teacher.load_from_checkpoint(pretrained_weights_teacher)
        
        if pretrained_weights_student is not None:
            self.student.load_from_checkpoint(pretrained_weights_student)

        if pretrained:
            for param in self.teacher.parameters():
                param.requires_grad = False

        self.hint_layers = [1,2,3]  
        self.training_stage = "hint"  

        metrics = MetricCollection({
            "minADE1": minADE(k=1),
            "minADE6": minADE(k=6),
            "minFDE1": minFDE(k=1),
            "minFDE6": minFDE(k=6),
            "MR": MR(),
            "brier-minFDE6": brierMinFDE(k=6)
        })
        self.val_metrics = metrics.clone(prefix="val_")
        self.curr_ep = 0

    def forward(self, data):
        return self.student(data)

    def predict(self, data, full=False):
        with torch.no_grad():
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

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2].contiguous(), y.contiguous())
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

    def diversity_loss(self, trajs: torch.Tensor) -> torch.Tensor:
        B, K, T, D = trajs.shape
        loss = 0.0
        count = 0
        for i in range(K):
            for j in range(i + 1, K):
                dist = torch.norm(trajs[:, i] - trajs[:, j], dim=-1).mean(dim=1)
                loss += (1.0 / (dist + 1e-6)).mean()
                count += 1
        return loss / count

    def hint_stage_training_step(self, data, batch_idx):
        self.student.train()

        for name, param in self.student.named_parameters():
            if "head" in name or "predictor" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        loss_hint = 0.0

        student_feat = self.student.get_guided_features(data)

        # Loop over all hint/guided layer pairs
        for i, layer_idx in enumerate(self.hint_layers):
            with torch.no_grad():
                teacher_feat = self.teacher.get_hint_features(data, layer_index=layer_idx)

            student_mapped = self.student.adapters[i](student_feat)

            min_len = min(student_mapped.size(1), teacher_feat.size(1))
            student_mapped = student_mapped[:, :min_len]
            teacher_feat = teacher_feat[:, :min_len]

            student_mapped = F.normalize(student_mapped, dim=-1)
            teacher_feat = F.normalize(teacher_feat, dim=-1)

            loss_hint += F.mse_loss(student_mapped, teacher_feat)

        loss_hint /= len(self.hint_layers)

        self.log("hint_stage/loss_hint", loss_hint.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss_hint

    def main_training_step(self, data, batch_idx):
        for param in self.student.parameters():
            param.requires_grad = True

        student_out = self.student(data)
        teacher_out = self.teacher(data) if self.teacher is not None else None

        losses = self.cal_loss(student_out, data)
        total_loss = losses["loss"]

        div_loss = self.diversity_loss(student_out["y_hat"])  

        hint_loss = 0.0
        if self.teacher is not None and hasattr(self.student, "adapters"):
            student_feat = self.student.get_guided_features(data)
            for i, layer_idx in enumerate(self.hint_layers):
                with torch.no_grad():
                    teacher_feat = self.teacher.get_hint_features(data, layer_index=layer_idx)

                student_mapped = self.student.adapters[i](student_feat)
                min_len = min(student_mapped.size(1), teacher_feat.size(1))
                student_mapped = student_mapped[:, :min_len]
                teacher_feat = teacher_feat[:, :min_len]

                student_mapped = F.normalize(student_mapped, dim=-1)
                teacher_feat = F.normalize(teacher_feat, dim=-1)

                hint_loss += F.mse_loss(student_mapped, teacher_feat)
            hint_loss /= len(self.hint_layers)
            total_loss += 0.05 * hint_loss  

        if self.teacher is not None:
            tau = 3.0
            # epoch = self.trainer.current_epoch if self.trainer else 0
            lambda_kd = 0.5


            s_logits = student_out["logits"]
            t_logits = teacher_out["logits"]

            s_probs = F.log_softmax(s_logits / tau, dim=-1)
            t_probs = F.softmax(t_logits / tau, dim=-1)

            kd_loss = F.kl_div(s_probs, t_probs, reduction="batchmean") * (tau ** 2)
            total_loss = (1 - lambda_kd) * total_loss + lambda_kd * kd_loss + 0.02 * div_loss

            self.log("train/kd_loss", kd_loss.item(), on_step=True, on_epoch=True)
            self.log("train/hint_loss", hint_loss.item(), on_step=True, on_epoch=True)

        self.log("train/total_loss", total_loss.item(), on_step=True, on_epoch=True)
        for k, v in losses.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True)

        return total_loss



    # switches from hint training to main training after a certain num epochs defined in config
    def training_step(self, data, batch_idx):
        if self.training_stage == "hint":
            return self.hint_stage_training_step(data, batch_idx)
        return self.main_training_step(data, batch_idx)

    def switch_to_main_training(self):
        self.training_stage = "main"
        self.student.train()

    def validation_step(self, data, batch_idx):
        student_out = self.student(data)
        losses = self.cal_loss(student_out, data, -1)
        metrics = self.val_metrics(student_out, data["y"][:, 0])

        self.log("val/reg_loss", losses["reg_loss"], on_step=False, on_epoch=True, prog_bar=False)
        for k in self.val_scores.keys():
            self.val_scores[k].append(metrics[k].item())
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        for i, layer_idx in enumerate(self.hint_layers):
            with torch.no_grad():
                teacher_feat = self.teacher.get_hint_features(data, layer_index=layer_idx)
                student_feat = self.student.get_guided_features(data)
                student_mapped = self.student.adapters[i](student_feat)

                min_len = min(student_mapped.size(1), teacher_feat.size(1))
                teacher_feat = F.normalize(teacher_feat[:, :min_len], dim=-1)
                student_mapped = F.normalize(student_mapped[:, :min_len], dim=-1)

                val_hint_loss = F.mse_loss(student_mapped, teacher_feat)
                self.log("val/hint_loss", val_hint_loss.item(), on_step=False, on_epoch=True)

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)

    def test_step(self, data, batch_idx) -> None:
        # out = self(data)
        out = self.student(data)
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