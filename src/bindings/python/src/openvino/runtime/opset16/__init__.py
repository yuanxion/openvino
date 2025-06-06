# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime.opset16 import ops

# New operations added in Opset16
from openvino.opset16.ops import identity
from openvino.opset16.ops import istft
from openvino.opset16.ops import segment_max
from openvino.opset16.ops import sparse_fill_empty_rows

# Operators from previous opsets
# TODO (ticket: 156877): Add previous opset operators at the end of opset16 development
