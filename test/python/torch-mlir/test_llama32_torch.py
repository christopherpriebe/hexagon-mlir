# ===- test_llama32_torch.py -----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
"""Llama 3.2 1B end-to-end test.

Gated on HuggingFace — set HF_TOKEN (or use `huggingface-cli login`)
before running. Truncated to 2 hidden layers; override via
`HEX_LLAMA32_LAYERS=N`. Override the model via `HEX_LLAMA32_MODEL`
(useful for swapping in `meta-llama/Llama-3.2-3B`, etc.).
"""

import os
import pytest
import torch

pytest.importorskip("transformers")
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import utils
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions


MODEL_NAME = os.environ.get("HEX_LLAMA32_MODEL", "meta-llama/Llama-3.2-1B")


def test_llama32():
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        pytest.skip("set HF_TOKEN to download gated Llama 3.2")
    n_layer = int(os.environ.get("HEX_LLAMA32_LAYERS", "2"))

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
