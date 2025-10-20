# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy as np

np.random.seed(42)

# arr1 = np.random.rand(32, 8, 512, 2048)


# arr1 = np.random.rand(20, 128, 32, 128)

# np_index = paddle.randint(0, 32, [32], dtype=paddle.int32)
np_index = paddle.randint(0, 32, [32], dtype=paddle.int32)


index_x = paddle.to_tensor(np_index.numpy(), dtype="int32")
index_y = paddle.to_tensor(np_index.numpy(), dtype="int32")

arr1 = np.random.rand(32, 512, 2048)
index_x = paddle.to_tensor([32], dtype="int32")
index_y = paddle.to_tensor([32], dtype="int32")

# Bf16
x_bf16 = paddle.to_tensor(arr1, dtype="bfloat16")

# Fp8
x_fp8 = paddle.to_tensor(arr1, dtype="bfloat16").astype(paddle.float8_e4m3fn)

import paddle.profiler as profiler

with profiler.Profiler(
    targets=[
        profiler.ProfilerTarget.CPU,
        profiler.ProfilerTarget.CUSTOM_DEVICE,
    ],
    scheduler=(10, 15),
    on_trace_ready=profiler.export_chrome_tracing("./log"),
) as p:
    for i in range(20):
        out1 = paddle.index_select(x=x_bf16, index=index_x, axis=0)
        out2 = paddle.index_select(x=x_fp8, index=index_y, axis=0)
        p.step()

# print(out)
