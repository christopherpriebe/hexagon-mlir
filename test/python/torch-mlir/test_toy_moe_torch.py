# ===- test_toy_moe_torch.py -----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
"""Hand-coded toy mixture-of-experts.

4 experts × small MLP each, top-2 gating with softmax over the gate
logits. Designed so the routing has STATIC shapes — torch.export can't
trace data-dependent shapes, and most real MoE implementations rely on
them (the per-token expert dispatch fan-out is dynamic). Here we sum
all experts weighted by their gate softmax, which keeps the trace
fully static at the cost of being closer to a "dense mixture" than a
sparse routing.

This exercises the IR shape of MoE — multiple parallel expert FFNs +
a gating linear + a weighted sum — without needing a real model load.
"""

import pytest
import torch
import torch.nn as nn

import utils
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions


class ToyMoE(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 512, n_experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts_w1 = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.experts_w2 = nn.Parameter(torch.empty(n_experts, hidden, dim))
        nn.init.normal_(self.experts_w1, std=0.02)
        nn.init.normal_(self.experts_w2, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, dim)
        gate_logits = self.gate(x)                   # (b, s, n_experts)
        gate_w = torch.softmax(gate_logits, dim=-1)  # (b, s, n_experts)

        # Apply each expert to x and weight by its gate value, then sum.
        # einsum: experts_w1[e, d, h], x[b, s, d] -> (b, s, e, h)
        h = torch.einsum("edh,bsd->bseh", self.experts_w1, x)
        h = torch.relu(h)
        # experts_w2[e, h, d], h[b, s, e, h] -> (b, s, e, d)
        y = torch.einsum("ehd,bseh->bsed", self.experts_w2, h)
        # Weight by gate: (b, s, e, 1) * (b, s, e, d) -> sum over e
        y = (gate_w.unsqueeze(-1) * y).sum(dim=2)    # (b, s, d)
        return y


def test_toy_moe():
    model = ToyMoE(dim=256, hidden=512, n_experts=4)
    model.eval()
    inp = torch.randn(1, 32, 256, dtype=torch.float32)

    manager = utils.ModelManager(model, inp)
    manager.create_linalg_module()
    manager.write_bytecode_to_file()
    manager.execute_and_compare(
        rtol=5e-3, atol=1e-5, dump_outputs=False,
        options=HexagonOptions().__dict__,
    )
