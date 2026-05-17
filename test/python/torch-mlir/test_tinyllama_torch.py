# ===- test_tinyllama_torch.py ---------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
"""TinyLlama 1.1B end-to-end test.

Open Llama-architecture model (no HF auth required). Truncated to 2
hidden layers via the HF config to keep the lowering tractable for
single-iteration tests — the full 22-layer model produces multi-GB
MLIR which is impractical for the splice-study capture loop. Set
`HEX_TINYLLAMA_LAYERS=N` to override.

Like the other Torch tests, on hosts without a Hexagon NPU the device
dispatch fails after compile, but the per-pass IR dumps are captured
beforehand.
"""

import os
import pytest
import torch

pytest.importorskip("transformers")
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import utils
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_tinyllama():
    n_layer = int(os.environ.get("HEX_TINYLLAMA_LAYERS", "2"))
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    cfg.num_hidden_layers = n_layer
    cfg.torch_dtype = torch.float16

    # `from_config` avoids downloading the full pretrained weights; we
    # don't need them since we're only exercising the compile pipeline,
    # not numerical correctness.
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
    model.eval()
    # Pin to a fixed RNG so the model has deterministic weights.
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
