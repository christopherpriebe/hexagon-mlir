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


class _Expert(nn.Module):
    """One expert: a 2-layer MLP."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class ToyMoE(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 512, n_experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([_Expert(dim, hidden) for _ in range(n_experts)])
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, dim) ; we use a dense softmax-weighted mixture
        # rather than top-k routing because torch.export can't trace
        # data-dependent gather/scatter cleanly.
        gate_logits = self.gate(x)                    # (b, s, n_experts)
        gate_w = torch.softmax(gate_logits, dim=-1)   # (b, s, n_experts)

        # Run each expert, stack outputs, weight by gate, sum.
        per_expert_outs = [self.experts[i](x) for i in range(self.n_experts)]
        # Each is (b, s, dim). Stack along a new expert dim -> (b, s, e, dim).
        ys = torch.stack(per_expert_outs, dim=2)
        # Gate shape (b, s, e) -> (b, s, e, 1) for broadcast.
        y = (gate_w.unsqueeze(-1) * ys).sum(dim=2)    # (b, s, dim)
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
