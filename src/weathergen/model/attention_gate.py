# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional, Tuple

import torch
from torch import nn


class AttentionGate(nn.Module):
    """Query-dependent attention gate copied and adapted from gated_attention/modeling_qwen3.py."""

    def __init__(
        self,
        dim_embed: int,
        num_heads: int,
        head_dim: int,
        bias: bool = False,
        headwise_attn_output_gate: bool = False,
        elementwise_attn_output_gate: bool = False,
    ) -> None:
        super().__init__()

        if headwise_attn_output_gate and elementwise_attn_output_gate:
            msg = "Only one of headwise_attn_output_gate or elementwise_attn_output_gate can be True."
            raise ValueError(msg)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate

        proj_dim = num_heads * head_dim
        if self.headwise_attn_output_gate:
            proj_dim += num_heads
        elif self.elementwise_attn_output_gate:
            proj_dim *= 2

        self.q_proj = nn.Linear(dim_embed, proj_dim, bias=bias)

    def project(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project inputs to query states and return gating logits."""

        query_states = self.q_proj(hidden_states)
        prefix_shape = hidden_states.shape[:-1]
        query_states = query_states.reshape(*prefix_shape, self.num_heads, -1)
        gate_score = None

        if self.headwise_attn_output_gate:
            query_states, gate_score = torch.split(
                query_states, [self.head_dim, 1], dim=-1
            )
            gate_score = gate_score.reshape(*prefix_shape, self.num_heads, 1)
        elif self.elementwise_attn_output_gate:
            query_states, gate_score = torch.split(
                query_states, [self.head_dim, self.head_dim], dim=-1
            )
            gate_score = gate_score.reshape(*prefix_shape, self.num_heads, self.head_dim)

        query_states = query_states.reshape(*prefix_shape, self.num_heads, self.head_dim)
        return query_states, gate_score

    @staticmethod
    def apply_gate(attn_output: torch.Tensor, gate_score: Optional[torch.Tensor]) -> torch.Tensor:
        """Multiply attention outputs with the sigmoid gate if available."""

        if gate_score is None:
            return attn_output
        return attn_output * torch.sigmoid(gate_score)
