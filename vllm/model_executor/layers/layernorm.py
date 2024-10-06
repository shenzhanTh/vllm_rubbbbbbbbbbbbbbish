"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from vllm._C import ops
import triton
import triton.language as tl

# @triton.jit
# def rms_norm_kernel(x, weight, epsilon, output, residual, num_elements):
#     pid = tl.program_id(0)
#     block_size = 64
#     idx = pid * block_size + tl.arange(0, block_size, dtype=tl.int32)

#     # 使用掩码避免越界
#     mask = idx < num_elements

#     # 计算均值和方差
#     sum_square = tl.zeros((1,), dtype=tl.float32)
#     for i in range(block_size):
#         if mask[i]:
#             sum_square += x[idx[i]] ** 2

#     sum_square = tl.sum(sum_square)  # 聚合所有块的平方和
#     mean_square = sum_square / num_elements
#     variance = mean_square + epsilon
#     scale = 1/(variance**0.5)

#     # 应用 RMSNorm
#     for i in range(block_size):
#         if mask[i]:
#             normed_value = x[idx[i]] * scale
#             output[idx[i]] = normed_value * weight

#     # 添加残差
#     if residual is not None:
#         for i in range(block_size):
#             if mask[i]:
#                 output[idx[i]] += residual[idx[i]]
class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def _forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out
        # num_elements = x.numel()
        # # 使用 Triton 内核进行 RMSNorm
        # out = torch.empty_like(x)
        # grid = (num_elements + 255) // 256,
        # rms_norm_kernel[grid](x, self.weight.data, self.variance_epsilon, out, residual, num_elements)
        # if residual is not None:
        #     out += residual.to(out.dtype)
        # return out
