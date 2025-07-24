import torch
import torch.nn as nn

class FeatureAdapter(nn.Module):
    """
    Adapter to handle different embedding sizes between teacher and student
    """
    def __init__(self, student_dim: int, teacher_dim: int, adapter_type: str = "linear"):
        super(FeatureAdapter, self).__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.adapter_type = adapter_type
        
        if student_dim != teacher_dim:
            if adapter_type == "linear":
                self.adapter = nn.Linear(student_dim, teacher_dim)
            elif adapter_type == "mlp":
                hidden_dim = max(student_dim, teacher_dim) // 2
                self.adapter = nn.Sequential(
                    nn.Linear(student_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, teacher_dim)
                )
            else:
                raise ValueError(f"Unknown adapter type: {adapter_type}")
        else:
            self.adapter = nn.Identity()
    
    def forward(self, student_features: torch.Tensor) -> torch.Tensor:
        return self.adapter(student_features)
