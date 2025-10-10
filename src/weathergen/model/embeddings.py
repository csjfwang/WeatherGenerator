# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from weathergen.model.attention import MultiSelfAttentionHead
from weathergen.model.layers import MLP

# from weathergen.model.mlp import MLP
from weathergen.model.norms import RMSNorm
from weathergen.model.positional_encoding import positional_encoding_harmonic


########################################################


########################################################
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class SelectorTransformer2(nn.Module):
    def __init__(self, dim_in, dim_embed, num_blocks=2, num_heads=4, k_max=16, dropout_rate=0.1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_embed = dim_embed
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.k_max = k_max

        self.norm = nn.LayerNorm(dim_embed)

        self.layers = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.layers.append(
                MultiSelfAttentionHead(
                    self.dim_embed,
                    self.num_heads,
                    dropout_rate=dropout_rate,
                    with_qk_lnorm=True,
                    with_flash=True,
                )
            )
            self.layers.append(
                MLP(
                    self.dim_embed,
                    self.dim_embed,
                    hidden_factor=2,
                    dropout_rate=dropout_rate,
                    with_residual=True,
                )
            )

        # scoring
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim_embed),
            nn.Linear(dim_embed, dim_embed),
            nn.GELU(),
            nn.Linear(dim_embed, 1)
        )

    def forward(self, x):
        B, N, D = x.shape
        x = self.norm(x)
        for layer in self.layers:
            x = layer(x)

        logits = self.scorer(x).squeeze(-1)  # [B, N]

        # Gumbel Top-K + STE
        U = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(U + 1e-9) + 1e-9)
        perturbed = logits + gumbel
        _, topk_idx = torch.topk(perturbed, self.k_max, dim=-1)

        hard_mask = torch.zeros_like(logits)
        hard_mask.scatter_(1, topk_idx, 1.0)
        soft = torch.softmax(logits, dim=-1)
        mask = (hard_mask - soft).detach() + soft

        # gather
        _, frame_topk = torch.topk(mask, self.k_max, dim=-1)
        idx_exp = frame_topk.unsqueeze(-1).expand(-1, -1, D)
        selected = torch.gather(x, 1, idx_exp)

        return selected
######


class SelectorTransformer(nn.Module):
    def __init__(self, dim_in, dim_embed, num_blocks=2, num_heads=4, k_max=8, num_ctrl=4):
        super().__init__()
        self.dim_in = dim_in
        self.dim_embed = dim_embed
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.k_max = k_max
        self.num_ctrl = num_ctrl

        # 输入已经是嵌入后的张量，不需要embedding层
        self.norm = nn.LayerNorm(dim_embed)

        # lightweight transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embed, nhead=num_heads,
            dim_feedforward=dim_embed * 2, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

        # learnable CTRL tokens
        self.ctrl = nn.Parameter(torch.randn(num_ctrl, dim_embed))

        # scoring MLP
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim_embed),
            nn.Linear(dim_embed, dim_embed),
            nn.GELU(),
            nn.Linear(dim_embed, 1)
        )

    @staticmethod
    def gumbel_topk_st(logits, k, tau=1.0):
        """
        FFS Top-K selection with Gumbel-Max Straight-Through Estimator
        
        Args:
            logits: [B, N] - 原始logits
            k: int - 要选择的top-k数量
            tau: float - 温度参数，控制Gumbel噪声的强度
            
        Returns:
            mask: [B, N] - 二进制mask，1表示被选中，0表示未选中
        """
        # 1. 添加Gumbel(0,1)噪声到logits
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + 1e-9) + 1e-9)
        perturbed_logits = logits + gumbel_noise
        
        # 2. 在perturbed logits上进行top-k选择
        # 使用argmax的离散选择（非可微分）
        _, topk_indices = torch.topk(perturbed_logits, k, dim=-1)
        
        # 3. 创建二进制mask
        binary_mask = torch.zeros_like(logits)
        binary_mask.scatter_(1, topk_indices, 1.0)
        
        # 4. Straight-Through Estimator (STE)
        # Forward pass: 使用离散的二进制mask
        # Backward pass: 梯度通过原始的logits流动
        ste_mask = binary_mask.detach() + logits - logits.detach()
        
        return ste_mask

    # 在 selector 内
    def forward(self, x):
        B, N, D = x.shape
        x = self.encoder(self.norm(x))
        ctrl = self.ctrl.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([x, ctrl], dim=1)             # [B, N+C, D]
        logits = self.scorer(tokens).squeeze(-1)         # [B, N+C]

        # Gumbel + Top-k + STE
        U = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(U + 1e-9) + 1e-9)
        perturbed = logits + gumbel
        topk_vals, topk_idx = torch.topk(perturbed, self.k_max, dim=-1)

        # hard mask + soft gradient
        hard_mask = torch.zeros_like(logits)
        hard_mask.scatter_(1, topk_idx, 1.0)
        soft = torch.softmax(logits, dim=-1)
        mask = (hard_mask - soft).detach() + soft         # STE 形式

        # 只取原始帧部分并 gather
        frame_mask = mask[:, :N]
        _, frame_topk = torch.topk(frame_mask, self.k_max, dim=-1)
        idx_exp = frame_topk.unsqueeze(-1).expand(-1, -1, D)
        selected_frames = torch.gather(x, 1, idx_exp)    # [B, k_max, D]
        return selected_frames
        # 在 selector 内
    # def forward(self, x):
    #     B, N, D = x.shape
    #     x = self.encoder(self.norm(x))
    #     ctrl = self.ctrl.unsqueeze(0).expand(B, -1, -1)
    #     tokens = torch.cat([x, ctrl], dim=1)             # [B, N+C, D]
    #     logits = self.scorer(tokens).squeeze(-1)         # [B, N+C]

    #     # Gumbel + Top-k + STE
    #     U = torch.rand_like(logits)
    #     gumbel = -torch.log(-torch.log(U + 1e-9) + 1e-9)
    #     perturbed = logits + gumbel
    #     topk_vals, topk_idx = torch.topk(perturbed, self.k_max, dim=-1)

    #     # hard mask + soft gradient
    #     hard_mask = torch.zeros_like(logits)
    #     hard_mask.scatter_(1, topk_idx, 1.0)
    #     soft = torch.softmax(logits, dim=-1)
    #     mask = (hard_mask - soft).detach() + soft         # STE 形式

    #     # 只取原始帧部分并 gather
    #     frame_mask = mask[:, :N]
    #     _, frame_topk = torch.topk(frame_mask, self.k_max, dim=-1)
    #     idx_exp = frame_topk.unsqueeze(-1).expand(-1, -1, D)
    #     selected_frames = torch.gather(x, 1, idx_exp)    # [B, k_max, D]
    #     return selected_frames
    # def forward(self, x):
    #     """
    #     x: [B, N, dim_embed] - 输入已经是嵌入后的张量
    #     """
    #     B, N, D = x.shape
        
    #     # 输入已经是嵌入后的，直接进行归一化和transformer编码
    #     x = self.norm(x)
    #     x = self.encoder(x)  # [B, N, D]
        
    #     # append CTRL tokens
    #     ctrl = self.ctrl.unsqueeze(0).expand(B, -1, -1)
    #     tokens = torch.cat([x, ctrl], dim=1)  # [B, N+C, D]
        
    #     # scoring - 对所有tokens (包括CTRL tokens) 进行打分
    #     scores = self.scorer(tokens).squeeze(-1)  # [B, N+C]
        
    #     # 只对原始frames应用选择，不包括CTRL tokens
    #     frame_scores = scores[:, :N]  # [B, N]
        
    #     # FFS Top-K选择：使用Gumbel-Max STE实现可微分的top-k选择
    #     # 1. 添加Gumbel噪声到logits
    #     U = torch.rand_like(frame_scores)
    #     gumbel_noise = -torch.log(-torch.log(U + 1e-9) + 1e-9)
    #     perturbed_scores = frame_scores + gumbel_noise
        
    #     # 2. 在perturbed scores上进行top-k选择（离散，非可微分）
    #     _, topk_indices = torch.topk(perturbed_scores, self.k_max, dim=-1)  # [B, k_max]
        
    #     # 3. 创建二进制mask
    #     binary_mask = torch.zeros_like(frame_scores)
    #     binary_mask.scatter_(1, topk_indices, 1.0)  # [B, N]
        
    #     # 4. Straight-Through Estimator (STE)
    #     # Forward pass: 使用离散的二进制mask
    #     # Backward pass: 梯度通过原始的frame_scores流动
    #     ste_mask = binary_mask.detach() + frame_scores - frame_scores.detach()
        
    #     # 5. 使用STE mask进行加权选择，保持可微分性
    #     selected_frames = ste_mask.unsqueeze(-1) * x  # [B, N, D]
        
    #     # 6. 只返回前k_max个frames（真正减少token数量）
    #     selected_frames = selected_frames[:, :self.k_max, :]  # [B, k_max, D]
        
    #     return selected_frames
    # def forward(self, x):
    #     """
    #     x: [B, N, dim_embed] - 输入已经是嵌入后的张量
    #     """
    #     B, N, _ = x.shape

    #     # 输入已经是嵌入后的，直接进行归一化和transformer编码
    #     x = self.norm(x)
    #     x = self.encoder(x)  # [B, N, D]

    #     # append CTRL tokens
    #     ctrl = self.ctrl.unsqueeze(0).expand(B, -1, -1)
    #     tokens = torch.cat([x, ctrl], dim=1)  # [B, N+C, D]

    #     # scoring - 对所有tokens (包括CTRL tokens) 进行打分
    #     scores = self.scorer(tokens).squeeze(-1)  # [B, N+C]
        
    #     # FFS Top-K selection with Gumbel-Max STE
    #     mask = self.gumbel_topk_st(scores, self.k_max)  # [B, N+C]
        
    #     # 只对原始frames应用mask，不包括CTRL tokens
    #     frame_mask = mask[:, :N]  # [B, N]
        
    #     # FFS Top-K选择：真正减少token数量
    #     # 使用可微分的top-k selection
        
    #     # 方法：使用Gumbel-TopK的soft selection，然后通过加权平均来减少token数量
    #     # 这样可以保持可微分性，同时实现真正的token数量减少
        
    #     # 获取frame scores（不包括CTRL tokens）
    #     frame_scores = scores[:, :N]  # [B, N]
        
    #     # 使用Gumbel-TopK获取soft weights
    #     soft_weights = self.gumbel_topk_st(frame_scores, self.k_max)  # [B, N]
        
    #     # FFS Top-K选择：真正符合原文的实现
    #     # 原文：前向使用离散的argmax top-k选择，反向梯度通过原始logits流动
        
    #     # 1. 添加Gumbel噪声到logits
    #     U = torch.rand_like(frame_scores)
    #     gumbel_noise = -torch.log(-torch.log(U + 1e-9) + 1e-9)
    #     perturbed_scores = frame_scores + gumbel_noise
        
    #     # 2. 在perturbed scores上进行top-k选择（离散，非可微分）
    #     _, topk_indices = torch.topk(perturbed_scores, self.k_max, dim=-1)  # [B, k_max]
        
    #     # 3. 创建二进制mask
    #     binary_mask = torch.zeros_like(frame_scores)
    #     binary_mask.scatter_(1, topk_indices, 1.0)  # [B, N]
        
    #     # 4. Straight-Through Estimator (STE)
    #     # Forward pass: 使用离散的二进制mask
    #     # Backward pass: 梯度通过原始的frame_scores流动
    #     ste_mask = binary_mask.detach() + frame_scores - frame_scores.detach()
        
    #     # 5. 应用mask到frames
    #     masked_frames = ste_mask.unsqueeze(-1) * x  # [B, N, D]
        
    #     # 6. 为了真正减少token数量，我们使用gather操作
    #     # 但这里会破坏梯度流，所以我们需要一个更好的方法
        
    #     # 替代方案：使用加权平均来模拟top-k选择，保持可微分性
    #     # 将mask转换为soft weights
    #     soft_weights = ste_mask / (ste_mask.sum(dim=-1, keepdim=True) + 1e-8)
        
    #     # 使用加权平均创建representative frames
    #     selected_frames = torch.einsum('bn,bnd->bnd', soft_weights, x)  # [B, N, D]
        
    #     # 只返回前k_max个frames
    #     selected_frames = selected_frames[:, :self.k_max, :]  # [B, k_max, D]
        
    #     return selected_frames
########################################################


class StreamEmbedTransformer(torch.nn.Module):
    def __init__(
        self,
        mode,
        num_tokens,
        token_size,
        num_channels,
        dim_embed,
        dim_out,
        num_blocks,
        num_heads,
        dropout_rate=0.0,
        norm_type="LayerNorm",
        embed_size_centroids=64,
        unembed_mode="full",
        stream_name="stream_embed",
    ):
        """Constructor

        unembed_mode : { 'full' , 'block'}
          full : monolithic (and correspondingly large) unembedding network that maps from
                 (num_tokens x dim_embed) to dim_out, allowing for mixing between channels/columns
          block : per-channel/column unembedding network
                (which is hence a block-sparse form of full)
        """

        super(StreamEmbedTransformer, self).__init__()

        self.name = f"StreamEmbedder_{stream_name}"

        self.num_tokens = num_tokens
        self.token_size = token_size
        self.num_channels = num_channels
        self.dim_in = token_size if mode == "channels" else num_channels
        self.dim_embed = dim_embed
        self.dim_out = dim_out
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.embed_size_centroids = embed_size_centroids
        self.unembed_mode = unembed_mode

        norm = torch.nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm

        self.channel_selection = True
        self.selector_mode = "fixed"   # 可选: "ctrl" / "fixed"

        if self.channel_selection:
            num_channels = 16

            if self.selector_mode == "ctrl":
                # 原来的 SelectorTransformer（带 CTRL tokens）
                self.selector = SelectorTransformer(
                    dim_in=self.dim_in,
                    dim_embed=self.dim_embed,
                    num_blocks=2,
                    num_heads=4,
                    k_max=num_channels,
                    num_ctrl=4
                )
            elif self.selector_mode == "fixed":
                # 新的 SelectorTransformer2（不带 CTRL，只选固定数量 channel）
                self.selector = SelectorTransformer2(
                    dim_in=self.dim_in,
                    dim_embed=self.dim_embed,
                    num_blocks=2,
                    num_heads=4,
                    k_max=num_channels
                )
            else:
                raise ValueError(f"Unknown selector_mode: {self.selector_mode}")

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_blocks):
            self.layers.append(
                MultiSelfAttentionHead(
                    self.dim_embed,
                    self.num_heads,
                    dropout_rate=dropout_rate,
                    with_qk_lnorm=True,
                    with_flash=True,
                )
            )
            self.layers.append(
                MLP(
                    self.dim_embed,
                    self.dim_embed,
                    hidden_factor=2,
                    dropout_rate=dropout_rate,
                    with_residual=True,
                )
            )

        if mode == "channels":
            self.embed = torch.nn.Linear(self.dim_in, self.dim_embed)

            if self.unembed_mode == "full":
                self.ln_final = norm(num_channels * self.dim_embed, eps=1e-03)
                self.unembed = torch.nn.Linear(
                    num_channels * self.dim_embed,
                    self.num_tokens * self.dim_out - embed_size_centroids,
                )

            elif self.unembed_mode == "block":
                # modify embed_size_centroids to ensure no additional padding is needed
                rem = (self.num_tokens * self.dim_out - embed_size_centroids) % num_channels
                embed_size_centroids += rem
                dim_out = (self.num_tokens * self.dim_out - embed_size_centroids) // num_channels
                self.unembed = torch.nn.ModuleList(
                    [torch.nn.Linear(dim_embed, dim_out) for _ in range(num_channels)]
                    # [
                    #     torch.nn.Sequential(
                    #         torch.nn.Linear(dim_embed, max(dim_embed//2,4*dim_out)),
                    #         torch.nn.GELU(),
                    #         torch.nn.Linear(max(dim_embed//2,4*dim_out), dim_out)
                    #     ) for _ in range(num_channels)
                    # ]
                )
                self.ln_final = torch.nn.ModuleList(
                    [norm(dim_embed, eps=1e-6) for _ in range(num_channels)]
                )

            else:
                assert False

            self.forward = self.forward_channels

        elif mode == "columns":
            assert embed_size_centroids == 0
            self.embed = torch.nn.Linear(self.dim_in, self.dim_embed)

            assert self.unembed_mode == "block"  # only supported mode at the moment
            # padding needed if the unembedded columns cannot be concatenated to dim_out (e.g GPSRO)
            self.pad = self.dim_out % token_size
            self.out_pad = torch.nn.Parameter(torch.zeros(self.pad))
            self.unembed = torch.nn.Linear(
                self.dim_embed,
                self.num_tokens * ((self.dim_out - embed_size_centroids) // token_size),
            )
            self.ln_final = norm(dim_out, eps=1e-6)
            self.forward = self.forward_columns

            # TODO: factorization when sqrt is not int
            dim1 = int(np.sqrt(dim_out))
            assert dim1 * dim1 == dim_out
            self.unembed1 = torch.nn.Linear(self.dim_embed, dim1)
            self.unembed_nonlin = torch.nn.GELU()
            self.unembed2 = torch.nn.Linear(self.token_size, dim1)

        else:
            assert False

        self.dropout_final = torch.nn.Dropout(0.1)
        self.embed_centroids = torch.nn.Linear(5, embed_size_centroids)

    def forward_channels(self, x_in, centroids):

        peh = positional_encoding_harmonic

        if self.channel_selection: 
            x_emded = checkpoint(self.embed, x_in.transpose(-2, -1), use_reentrant=False)
            selected = self.selector(x_emded[:, 3:74, :].contiguous())
            # embed provided input data
            x = peh(selected)  # [B, k_max, D]

            # x_emded = peh(checkpoint(self.embed, x_in.transpose(-2, -1), use_reentrant=False))
            # # x = peh(x_emded)
            # selected = self.selector(x_emded[:, 3:74, :].contiguous())
            # x = torch.cat([x_emded[:, :3, :], selected, x_emded[:, 74:, :]], dim=1)
        else:
            # embed provided input data
            x = peh(checkpoint(self.embed, x_in.transpose(-2, -1), use_reentrant=False))


        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)

        # read out
        if self.unembed_mode == "full":
            out = checkpoint(self.unembed, self.ln_final(x.flatten(-2, -1)), use_reentrant=False)
        elif self.unembed_mode == "block":
            out = [
                checkpoint(ue, ln(x[:, i]), use_reentrant=False)
                for i, (ue, ln) in enumerate(zip(self.unembed, self.ln_final, strict=True))
            ]
            out = torch.stack(out, dim=1).flatten(-2, -1)
        else:
            assert False

        # append centroids
        if self.embed_size_centroids > 0:
            out = torch.cat([out, self.embed_centroids(centroids)], -1)
        # if self.embed_size_centroids==0 and self.dim_out is not divisible by #channels with
        # unembed_mode block then we need to pad to have the expected output shape
        if out.shape[-1] < self.dim_out:
            out = torch.nn.functional.pad(out, [0, self.dim_out - out.shape[-1]], value=0.0)
        # final reshape
        out = self.dropout_final(out.reshape(-1, self.num_tokens, self.dim_out))

        return out

    def forward_columns(self, x_in, centroids):
        # embed provided input data
        x = positional_encoding_harmonic(checkpoint(self.embed, x_in, use_reentrant=False))

        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)

        out = checkpoint(self.unembed1, x, use_reentrant=False)
        out = self.unembed_nonlin(out)
        out = checkpoint(self.unembed2, out.transpose(-2, -1), use_reentrant=False)
        out = out.flatten(-2, -1).unsqueeze(1)

        # final normalize and dropout
        out = self.dropout_final(self.ln_final(out))

        return out.to(torch.float16)


class StreamEmbedLinear(torch.nn.Module):
    def __init__(self, dim_in, dim_out, stream_name="stream_embed"):
        """Constructor"""

        super(StreamEmbedLinear, self).__init__()

        self.name = f"StreamEmbedder_{stream_name}"
        self.layer = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x):
        # x = checkpoint( self.layer, x.flatten( -2, -1), use_reentrant=True)
        x = self.layer(x.flatten(-2, -1))

        return x
