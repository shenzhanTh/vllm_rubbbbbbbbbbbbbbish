"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from vllm._C import ops
import triton
import triton.language as tl


# @triton.jit
# def rms_norm_kernel(x, weight, epsilon, output, residual, num_elements, BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(0)
#     idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

#     # 使用掩码避免越界
#     mask = idx < num_elements

#     # 计算均值和方差
#     sum_square = tl.zeros((1,), dtype=tl.float32)
#     count = tl.zeros((1,), dtype=tl.int32)

#     # 计算平方和
#     square_values = tl.where(mask, x[idx] ** 2, 0.0)
#     sum_square += tl.sum(square_values)
#     count += tl.sum(mask)

#     # 使用一个线程计算均值和方差
#     if pid == 0:  # 只有一个线程计算均值和方差
#         mean_square = sum_square / count
#         variance = mean_square + epsilon
#         scale = 1/tl.sqrt(variance)

#         # 应用 RMSNorm
#         output[idx] = tl.where(mask, x[idx] * scale * weight, 0.0)

#         # 添加残差
#         if residual is not None:
#             output[idx] += tl.where(mask, residual[idx], 0.0)
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
        # BLOCK_SIZE = 64
        # rms_norm_kernel[(grid,BLOCK_SIZE)](x, self.weight.data, self.variance_epsilon, out, residual, num_elements,BLOCK_SIZE)
        # if residual is not None:
        #     out += residual.to(out.dtype)
        # return out
