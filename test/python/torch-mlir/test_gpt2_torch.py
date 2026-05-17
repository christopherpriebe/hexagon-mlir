# ===- test_gpt2_torch.py ---------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
"""GPT-2 end-to-end test.

Thin pytest wrapper around the existing gpt2lmheadmodel.py driver, which
loads openai-community/gpt2 (truncated to n_layer=2 for tractability),
exports it via torch-mlir, and runs it through the Hexagon backend.

On hosts without a Hexagon NPU the device dispatch fails after the compile,
but the per-pass IR dumps (when MLIR_ENABLE_DUMP=1) are already captured.
"""

import pytest

pytest.importorskip("transformers")

from gpt2lmheadmodel import gpt2lmheadmodel


def test_gpt2():
    gpt2lmheadmodel()
