# ===- test_qwen_moe_torch.py ----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
"""Qwen 1.5 MoE A2.7B end-to-end test.

**Currently skipped.** The HuggingFace `transformers` implementation
of Qwen2-MoE (`modeling_qwen2_moe.py:613`) does:

    expert_hitted = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero(
        as_tuple=True)[0].tolist()

The `.nonzero().tolist()` chain produces a Python list whose length
depends on the runtime tensor values (how many experts got routed to).
`torch.export` is a static-shape tracer and can't lower this:

    torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
    Could not guard on data-dependent expression Eq(u0, 0)

Set `HEX_QWEN_MOE_FORCE=1` to attempt the export anyway (it will fail).
Mixtral, DeepSeek-V2-MoE, and the other production MoEs in transformers
have the same pattern, so swapping models doesn't help. For MoE-shape
IR coverage, see `test_toy_moe_torch.py` which uses a dense
softmax-weighted mixture that traces cleanly.

When transformers ships a torch.export-friendly MoE implementation (or
the LLM-Compiler / Optimum patches land), drop this skip.
"""

import os
import pytest
import torch

pytest.importorskip("transformers")
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import utils
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions


MODEL_NAME = os.environ.get("HEX_QWEN_MOE_MODEL", "Qwen/Qwen1.5-MoE-A2.7B")


def test_qwen_moe():
    if os.environ.get("HEX_QWEN_MOE_FORCE", "0") != "1":
        pytest.skip(
            "Qwen MoE's transformers implementation uses .nonzero().tolist() "
            "for expert routing, which torch.export rejects as data-dependent. "
            "Set HEX_QWEN_MOE_FORCE=1 to attempt the export anyway (will fail "
            "with GuardOnDataDependentSymNode). See test_toy_moe_torch.py for "
            "MoE-shape IR coverage that does trace."
        )
    n_layer = int(os.environ.get("HEX_QWEN_MOE_LAYERS", "1"))
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    cfg.num_hidden_layers = n_layer
    cfg.torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
    model.eval()
    torch.manual_seed(0)
    for p in model.parameters():
        p.data.normal_(mean=0.0, std=0.02)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoding = tokenizer("What is the nature of our existence?", return_tensors="pt")
    input_ids = encoding["input_ids"]

    manager = utils.ModelManager(model, input_ids)
    manager.create_linalg_module()
    manager.write_bytecode_to_file()
    manager.execute_and_compare(
        rtol=5e-3, atol=1e-5, dump_outputs=False,
        options=HexagonOptions().__dict__,
    )
