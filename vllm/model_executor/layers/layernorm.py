"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from vllm._C import ops
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, output_ptr, n_elements, variance_epsilon, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Load input tensor x
    x = tl.load(tl.make_tensor(x_ptr + offsets, dtype=tl.float32), mask=mask)
    
    # Compute the mean of squares
    squared = x * x
    mean_squared = tl.sum(squared, axis=0) / n_elements
    
    # Compute the RMS
    rms = tl.sqrt(mean_squared + variance_epsilon)
    
    # Normalize
    normalized = (x / rms) * tl.load(weight_ptr)

    # Store the result
    tl.store(output_ptr + offsets, normalized, mask=mask)

def rms_norm(x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    # 我们需要预先分配输出。
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    
    n_elements = output.numel()
    
    # 定义启动网格
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # 调用 Triton 内核
    rms_norm_kernel[grid](x.data_ptr(), weight.data_ptr(), output.data_ptr(), n_elements, variance_epsilon, BLOCK_SIZE=256)
    
    return output

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
        # if residual is not None:
        #     ops.fused_add_rms_norm(
        #         x,
        #         residual,
        #         self.weight.data,
        #         self.variance_epsilon,
        #     )
        #     return x, residual
        # out = torch.empty_like(x)
        # ops.rms_norm(
        #     out,
        #     x,
        #     self.weight.data,
        #     self.variance_epsilon,
        # )
        # return out
        output = rms_norm(x, self.weight, self.variance_epsilon)
        if residual is not None:
            output += residual.to(output.dtype)
        return output
