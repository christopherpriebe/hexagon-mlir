# ===- test_qwen_moe_torch.py ----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
"""Qwen 1.5 MoE A2.7B end-to-end test.

Open MoE LLM from Alibaba. ~14 GB on disk at fp16. Truncate via
`HEX_QWEN_MOE_LAYERS=N` (default 1 — even one layer of an MoE produces
substantial IR). The real model has 24 layers, 60 experts, top-4 routing.

Note: torch.export traces a STATIC computation graph. Real MoEs route
each token to a sparse subset of experts at runtime, which torch.export
struggles to lower without scripting tricks. The Qwen MoE implementation
in transformers does a per-token gather; on export this typically
materializes as a static dense-style dispatch. The resulting IR is
still illustrative of the MoE pipeline shape, even if not exactly the
runtime behavior.
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
