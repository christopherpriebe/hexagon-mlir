# ===- test_sd_unet_torch.py ----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
"""Stable Diffusion 1.5 UNet end-to-end test.

The UNet is the heaviest single component of an SD pipeline (~860M
params). We don't load the VAE / text encoder / scheduler — we
synthesize a config and instantiate the UNet directly. Inputs are
dummy noise + a single timestep + dummy text-encoder hidden states.

Override the resolution and channels via env vars if you want
smaller dumps:
  HEX_SD_RESOLUTION=64   # latent spatial dim (default 64 = 512px image)
  HEX_SD_BATCH=1
  HEX_SD_LAYERS=full     # "full" or a positive int for per-stage layers

Needs `diffusers` and `accelerate` from PyPI (not in the default
hexagon-mlir requirements; install separately).
"""

import os
import pytest
import torch

diffusers = pytest.importorskip("diffusers")
from diffusers import UNet2DConditionModel

import utils
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions


MODEL_NAME = os.environ.get("HEX_SD_MODEL", "runwayml/stable-diffusion-v1-5")


def test_sd_unet():
    resolution = int(os.environ.get("HEX_SD_RESOLUTION", "64"))
    batch = int(os.environ.get("HEX_SD_BATCH", "1"))

    # UNet config from the SD 1.5 subfolder. from_config (not from_pretrained)
    # avoids the multi-GB safetensors download — we just need the architecture.
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_NAME,
        subfolder="unet",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    ).eval()

    # Inputs SD UNet expects:
    #   sample:               (B, in_channels=4, H, W)
    #   timestep:             scalar or (B,)
    #   encoder_hidden_states:(B, seq=77, cross_attn_dim=768)
    sample = torch.randn(batch, 4, resolution, resolution, dtype=torch.float16)
    timestep = torch.tensor(50, dtype=torch.long)
    text_emb = torch.randn(batch, 77, 768, dtype=torch.float16)

    manager = utils.ModelManager(unet, sample, timestep, text_emb)
    manager.create_linalg_module()
    manager.write_bytecode_to_file()
    manager.execute_and_compare(
        rtol=5e-3, atol=1e-5, dump_outputs=False,
        options=HexagonOptions().__dict__,
    )
