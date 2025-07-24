from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath
from torch import Tensor
import math



"""
Quadratic attention formula:
    Attention(Q,K,V) = softmax(QK^T/√d)V
    
Linear attention formula:
    Attention(Q,K,V) = φ(Q)(φ(K)^T V) / (φ(Q)(φ(K)^T 1))
"""

class LinearAttention(nn.Module):
    """ Linear Attention Module for lightweight attention mechanism and reduced memory usage. """
    
    def __init__(
        self,
        dim,
        num_heads,
        drop=0.0,
        bias=True,
        act_layer=nn.GELU,
        kdim=None,
        vdim=None,
        feature_map='elu'
    ):
        
        # Get the parameters
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.vdim = vdim or dim
        self.kdim = kdim or dim
        self.drop = drop
        
        # Ensure that the dimension is divisible by the number of heads. This is because in multi head attention,
        # we split the imput embedding dimension into multiple heads
        assert self.head_dim * num_heads == self.dim, "dim must be divisible by num_heads"
        
        # Here we define the linear projections for query, key, and value
        self.q_proj = nn.Linear(self.dim, self.dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, self.dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, self.dim, bias=bias)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=bias)
        
        self.dropout = nn.Dropout(drop)
        
        """
            Feature map for kernel approximation
            The kernel function determines the similarity between the query and key vectors. In standard attention,
            we use softmax
            Feature maps are used to approximate the kernel function in linear attention.
            
            Instead of computing:
            kernel(q, k) = exp(q·k)

            We can compute:
            φ(q)^T @ φ(k) ≈ exp(q·k)
            
            This approximates the kernel using dot products in a different space
        """
        if feature_map == 'elu':
            self.feature_map = lambda x: F.elu(x) + 1
        elif feature_map == 'relu':
            self.feature_map = lambda x: F.relu(x)
        else:
            self.feature_map = lambda x: x
        
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None
    ):
        # φ(Q) @ (φ(K)^T @ V) 
        
        if key is None:
            key = query
        if value is None:
            value = key
        
        B, N, C = query.shape
        _, S, _ = key.shape
        
        # Project the query, key, and value tensors into the attention space
        q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # B, H, N, D
        k = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # B, H, S, D
        v = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # B, H, S, D
        
        # Apply feature map for linear attention
        q = self.feature_map(q) # Fi(Q) # B, H, N, D
        k = self.feature_map(k) # Fi(K) # B, H, S, D
        
        # Handle key padding mask - fix dimensions
        if key_padding_mask is not None:
            # key_padding_mask shape: [B, S]
            # k and v shape: [B, H, S, D] where H=num_heads, D=head_dim
            # We need to expand mask to [B, H, S, 1] to broadcast with [B, H, S, D]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, S, 1]
            mask = mask.expand(B, self.num_heads, S, 1)  # [B, H, S, 1]
            k = k.masked_fill(mask, 0.0) # [B, H, S, D]
            v = v.masked_fill(mask, 0.0) 
        
        # Linear attention computation O(N)
        # torch.einsum calculates the key-value product 
        # Note that in self attention D and E are the same, so we can use the same dimension for both
        kv = torch.einsum('bhsd,bhse->bhde', k, v)  # [B, H, D, E]
        attn_output = torch.einsum('bhqd,bhde->bhqe', q, kv)  # [B, H, Q, E]
        
        # Normalization - fix the tensor dimensions
        k_sum = k.sum(dim=2)  # [B, H, D] - sum over sequence dimension
        normalizer = torch.einsum('bhqd,bhd->bhq', q, k_sum)  # [B, H, Q]
        normalizer = normalizer.unsqueeze(-1) + 1e-6  # [B, H, Q, 1]
        attn_output = attn_output / normalizer
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)
        
        # Apply dropout and output projection
        attn_output = self.dropout(attn_output)
        return self.out_proj(attn_output)
    
class PerformerAttention(nn.Module):
    """ Performer Attention Module for efficient attention computation using kernel approximation. """
    
    def __init__(
        self,
        dim,
        num_heads,
        drop=0.0,
        bias=True,
        act_layer=nn.GELU,
        kdim=None,
        vdim=None,
        nb_feat=None
    ):
        
        # φ(Q) @ φ(K)^T @ V / normalization
        
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.vdim = vdim or dim
        self.kdim = kdim or dim
        self.drop = drop
        
        assert self.head_dim * num_heads == self.dim, "dim must be divisible by num_heads"
        
        self.nb_feat = nb_feat or max(32, int(self.head_dim * math.log(self.head_dim)))
        
        self.q_proj = nn.Linear(self.dim, self.dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, self.dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, self.dim, bias=bias)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=bias)
        
        self.dropout = nn.Dropout(drop)
        
        # Random features for FAVOR+
        self.register_buffer('random_features', 
                           torch.randn(self.nb_feat, self.head_dim))
        
    def kernel_feature_creator(self, x):
        """ Create kernel features for Performer attention. """
        # This function approximates exp(q·k / sqrt(d)) with φ(q)^T φ(k) ≈ exp(q·k / sqrt(d)) where φ 
        # uses random features to create a kernel approximation.
        # φ(x) = (1/√m) * exp(Ωx - ||x||²/2) * [cos(Ωx), sin(Ωx)]
        # We use this instead of a predefined feature map to allow for flexibility in the kernel approximation
        norm = 1.0 / math.sqrt(math.sqrt(self.head_dim)) # Normalization factor 1/ sqrt(sqrt(d))
        ratio = 1.0 / math.sqrt(self.nb_feat)
        
        x = x * norm
        x_dash = torch.einsum('...nd,md->...nm', x, self.random_features)
        x_diag = torch.sum(x ** 2, dim=-1, keepdim=True) / 2.0
        x_dash = ratio * torch.exp(x_dash - x_diag)
        
        return x_dash
    
    def forward(self, query, key=None, value=None, key_padding_mask=None, attn_mask=None):
        if key is None:
            key = query
        if value is None:
            value = key
            
        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply kernel feature mapping
        q = self.kernel_feature_creator(q)
        k = self.kernel_feature_creator(k)
        
        # Handle key padding mask - fix dimensions
        if key_padding_mask is not None:
            # key_padding_mask shape: [B, S]
            # k and v shape: [B, H, S, D] where H=num_heads, D=head_dim
            # We need to expand mask to [B, H, S, 1] to broadcast with [B, H, S, D]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, S, 1]
            mask = mask.expand(batch_size, self.num_heads, src_len, 1)  # [B, H, S, 1]
            k = k.masked_fill(mask, 0.0)
            v = v.masked_fill(mask, 0.0)
        
        # Linear attention computation
        kv = torch.einsum('bhsd,bhse->bhde', k, v)
        attn_output = torch.einsum('bhqd,bhde->bhqe', q, kv)
        
        # Normalization - fix the tensor dimensions
        k_sum = k.sum(dim=2)  # [B, H, D] - sum over sequence dimension
        normalizer = torch.einsum('bhqd,bhd->bhq', q, k_sum)  # [B, H, Q]
        normalizer = normalizer.unsqueeze(-1) + 1e-6  # [B, H, Q, 1]
        attn_output = attn_output / normalizer
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, embed_dim
        )
        
        # Apply dropout and output projection
        attn_output = self.dropout(attn_output)
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        post_norm=False,
        cross_attn=False,
        kdim=None,
        vdim=None,
        attention_type='standard'  # 'linear', 'performer', or 'standard'
    ):
        super().__init__()
        self.post_norm = post_norm
        self.cross_attn = cross_attn
        self.attention_type = attention_type  # Store attention type

        if kdim is None: kdim = dim
        if vdim is None: vdim = dim
        if self.cross_attn:
            if vdim is not None: self.normkv = norm_layer(vdim)
            if kdim is not None: self.normk = norm_layer(kdim)
        self.norm1 = norm_layer(dim)
        
        # Choose attention mechanism based on type
        if attention_type == 'linear':
            self.attn = LinearAttention(
                dim=dim,
                num_heads=num_heads,
                drop=attn_drop,
                bias=qkv_bias,
                kdim=kdim,
                vdim=vdim
            )
        elif attention_type == 'performer':
            self.attn = PerformerAttention(
                dim=dim,
                num_heads=num_heads,
                drop=attn_drop,
                bias=qkv_bias,
                kdim=kdim,
                vdim=vdim
            )
        else:  # standard attention
            self.attn = torch.nn.MultiheadAttention(
                dim,
                num_heads=num_heads,
                dropout=attn_drop,
                add_bias_kv=qkv_bias,
                batch_first=True,
                kdim=kdim,
                vdim=vdim
            )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_pre(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        if self.attention_type in ['linear', 'performer']:
            src2 = self.attn(
                query=src2,
                key=src2,
                value=src2,
                key_padding_mask=key_padding_mask,
                attn_mask=mask,
            )
        else:
            src2 = self.attn(
                query=src2,
                key=src2,
                value=src2,
                attn_mask=mask,
                key_padding_mask=key_padding_mask,
            )[0]
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src
    
    def forward_custom(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        kv: Optional[Tensor] = None,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
    ):
        assert (k is None and v is None) or (k is not None and v is not None)
        if k is not None:
            q = self.norm1(src)
            k = self.normk(k)
            v = self.normkv(v)
        elif kv is not None:
            q = self.norm1(src)
            k = v = self.normkv(kv)
        else:
            q = k = v = self.norm1(src)
            
        if self.attention_type in ['linear', 'performer']:
            attn_output = self.attn(
                query=q,
                key=k,
                value=v,
                key_padding_mask=key_padding_mask,
                attn_mask=mask,
            )
        else:
            attn_output = self.attn(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                key_padding_mask=key_padding_mask,
            )[0]
        src = q + self.drop_path1(attn_output) 
        src = src + self.drop_path2(self.mlp(self.norm2(src))) 
        return src
    
    def forward_post(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        if self.attention_type in ['linear', 'performer']:
            src2 = self.attn(
                query=src,
                key=src,
                value=src,
                key_padding_mask=key_padding_mask,
                attn_mask=mask,
            )
        else:
            src2 = self.attn(
                query=src,
                key=src,
                value=src,
                attn_mask=mask,
                key_padding_mask=key_padding_mask,
            )[0]
        src = src + self.drop_path1(self.norm1(src2))
        src = src + self.drop_path2(self.norm2(self.mlp(src)))
        return src

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        kv: Optional[Tensor] = None,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
    ):
        if self.cross_attn:
            return self.forward_custom(
                src=src, kv=kv, mask=mask, k=k, v=v, key_padding_mask=key_padding_mask
            )
        if self.post_norm:
            return self.forward_post(
                src=src, mask=mask, key_padding_mask=key_padding_mask
            )
        return self.forward_pre(src=src, mask=mask, key_padding_mask=key_padding_mask)
