from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from torch import distributed as dist


from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce, tensor_model_parallel_all_gather)
from vllm.model_executor.parallel_utils.utils import (
    divide, split_tensor_along_last_dim)
from vllm.model_executor.utils import set_weight_attrs
from vllm.logger import init_logger
import os


logger = init_logger(__name__)


def adjust_marlin_shard(param, shard_size, shard_offset):
    marlin_tile_size = getattr(param, "marlin_tile_size", None)
    if marlin_tile_size is None:
        return shard_size, shard_offset

    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


class LinearMethodBase(ABC):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(self, input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        """Create weights for a linear layer."""
        raise NotImplementedError

    @abstractmethod
    def apply_weights(self,
                      weights: Dict[str, torch.Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the weights to the input tensor."""
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization.

    Args:
        separate_bias_add: If true, add bias separately after matrix
                           multiplication.
    """

    def __init__(self, separate_bias_add: bool = False):
        self.separate_bias_add = separate_bias_add
        self.use_llama_nn = os.environ.get('LLAMA_NN') == '1'

    def create_weights(self, input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        weight = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        return {"weight": weight}

    def apply_weights(self,
                      weights: Dict[str, torch.Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = weights["weight"]
        if self.separate_bias_add:
            if bias:
                return F.linear(x, weight) + bias
            return F.linear(x, weight)
        
        if self.use_llama_nn:
            weight = weight.reshape(weight.shape[1], -1) ##权重重塑
            if bias is not None:
                return torch.matmul(x, weight) + bias
            else:
                return torch.matmul(x, weight) 
        else:
            return F.linear(x, weight, bias)
        """参数提取:
        从 weights 字典中提取权重 weight。
        分离偏置添加:
        如果 self.separate_bias_add 为 True,首先调用 F.linear 进行矩阵乘法(x @ weight),然后根据 bias 的存在与否决定是否加上偏置。
        使用 LLAMA NN:
        如果 self.use_llama_nn 为 True,权重重塑为 (output_size, -1)，以便与输入 x 进行矩阵乘法。
        根据 bias 的存在与否，选择是否加上偏置。
        默认情况:
        使用 PyTorch 的 F.linear 直接进行矩阵乘法和偏置的加法。"""


class ReplicatedLinear(torch.nn.Module):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        linear_method: (Maybe quantized) linear method.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        if linear_method is None:
            linear_method = UnquantizedLinearMethod()
        self.linear_method = linear_method
        self.linear_weights = self.linear_method.create_weights(
            self.input_size, self.output_size, self.input_size,
            self.output_size, self.params_dtype)
        for name, weight in self.linear_weights.items():
            if isinstance(weight, torch.Tensor):
                self.register_parameter(name, weight)
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype))
            set_weight_attrs(self.bias, {"output_dim": 0})
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if not self.skip_bias_add else None
        output = self.linear_method.apply_weights(self.linear_weights, x, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        linear_method: (Maybe quantized) linear method.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        tp_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, tp_size)
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        if linear_method is None:
            linear_method = UnquantizedLinearMethod()
        self.linear_method = linear_method
        self.linear_weights = self.linear_method.create_weights(
            self.input_size, self.output_size_per_partition, self.input_size,
            self.output_size, self.params_dtype)
        for name, weight in self.linear_weights.items():
            if isinstance(weight, torch.Tensor):
                self.register_parameter(name, weight)
                set_weight_attrs(weight, {"weight_loader": self.weight_loader})
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)
        self.use_llama_nn = os.environ.get('LLAMA_NN') == '1'

    # @torch.compile #######编译优化
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        output_dim = getattr(param, "output_dim", None)
        param_data = param.data
        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)
        assert param_data.shape == loaded_weight.shape
        if self.use_llama_nn:
            loaded_weight = loaded_weight.transpose(0, 1)
            loaded_weight = loaded_weight.reshape(param_data.shape[0],-1)
        param_data.copy_(loaded_weight)

    # @torch.compile #######编译优化
    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        output_parallel = self.linear_method.apply_weights(
            self.linear_weights, input_, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        linear_method: (Maybe quantized) linear method.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        self.output_sizes = output_sizes
        tp_size = get_tensor_model_parallel_world_size()
        assert all(output_size % tp_size == 0 for output_size in output_sizes)
        super().__init__(input_size, sum(output_sizes), bias, gather_output,
                         skip_bias_add, params_dtype, linear_method)
        self.use_llama_nn = os.environ.get('LLAMA_NN') == '1'

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        if loaded_shard_id is None:
            # Loaded weight is already packed.
            if output_dim is None:
                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            current_shard_offset = 0
            shard_offsets = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor

                    # If marlin, we need to adjust the offset and size to account for the tiling.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset)

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id < len(self.output_sizes)
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
            shard_size = self.output_sizes[loaded_shard_id] // tp_size
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor

                # If marlin, we need to adjust the offset and size to account for the tiling.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset)

            if self.use_llama_nn:
                param_data_ = param_data.narrow(output_dim, shard_offset,
                                            shard_size)
            else:
                param_data = param_data.narrow(output_dim, shard_offset,
                                            shard_size)
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions.")
        
        if self.use_llama_nn:
            assert param_data_.shape == loaded_weight.shape
            param_data_.copy_(loaded_weight)
            if loaded_shard_id == 1 and len(param_data.shape) == 2:
                param_data = param_data.transpose(0, 1)
                param.data = param_data.reshape(param_data.shape[1], -1)
        else:
            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)

        ###########################################################################
        # # 使用 no_grad 来避免不必要的梯度跟踪
        # with torch.no_grad():
        #     if loaded_shard_id is None:
        #         # Loaded weight is already packed.
        #         if output_dim is None:
        #             assert param_data.shape == loaded_weight.shape
        #             param_data.copy_(loaded_weight)
        #             return
        #         current_shard_offset = 0
        #         shard_offsets = []
        #         for i, output_size in enumerate(self.output_sizes):
        #             shard_offsets.append((i, current_shard_offset, output_size))
        #             current_shard_offset += output_size
        #         packed_dim = getattr(param, "packed_dim", None)
        #         for shard_id, shard_offset, shard_size in shard_offsets:
        #             # If quantized, we need to adjust the offset and size to account
        #             # for the packing.
        #             if packed_dim == output_dim:
        #                 shard_size = shard_size // param.pack_factor
        #                 shard_offset = shard_offset // param.pack_factor

        #                 # If marlin, we need to adjust the offset and size to account for the tiling.
        #                 shard_size, shard_offset = adjust_marlin_shard(
        #                     param, shard_size, shard_offset)

        #             loaded_weight_shard = loaded_weight.narrow(
        #                 output_dim, shard_offset, shard_size)
        #             self.weight_loader(param, loaded_weight_shard, shard_id)
        #         return
        #     """
        #     assert loaded_shard_id < len(self.output_sizes)
        #     tp_rank = get_tensor_model_parallel_rank()
        #     tp_size = get_tensor_model_parallel_world_size()
        #     if output_dim is not None:
        #         shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
        #         shard_size = self.output_sizes[loaded_shard_id] // tp_size
        #         # If quantized, we need to adjust the offset and size to account
        #         # for the packing.
        #         packed_dim = getattr(param, "packed_dim", None)
        #         if packed_dim == output_dim:
        #             shard_size = shard_size // param.pack_factor
        #             shard_offset = shard_offset // param.pack_factor

        #             # If marlin, we need to adjust the offset and size to account for the tiling.
        #             shard_size, shard_offset = adjust_marlin_shard(
        #                 param, shard_size, shard_offset)

        #         if self.use_llama_nn:
        #             param_data_ = param_data.narrow(output_dim, shard_offset,
        #                                         shard_size)
        #         else:
        #             param_data = param_data.narrow(output_dim, shard_offset,
        #                                         shard_size)
        #         start_idx = tp_rank * shard_size
        #         loaded_weight = loaded_weight.narrow(output_dim, start_idx,
        #                                             shard_size)
        #     else:
        #         ignore_warning = getattr(param, "ignore_warning", False)
        #         if not ignore_warning:
        #             logger.warning(
        #                 "Loading a weight without `output_dim` attribute in "
        #                 "MergedColumnParallelLinear, assume the weight is "
        #                 "the same for all partitions.")
            
        #     if self.use_llama_nn:
        #         assert param_data_.shape == loaded_weight.shape
        #         param_data_.copy_(loaded_weight)
        #         if loaded_shard_id == 1 and len(param_data.shape) == 2:
        #             param_data = param_data.transpose(0, 1)
        #             param.data = param_data.reshape(param_data.shape[1], -1)
        #     else:
        #         assert param_data.shape == loaded_weight.shape
        #         param_data.copy_(loaded_weight)"""
        #     # 处理 loaded_shard_id 的情况
        #     assert loaded_shard_id < len(self.output_sizes)
        #     tp_rank = get_tensor_model_parallel_rank()
        #     tp_size = get_tensor_model_parallel_world_size()

        #     if output_dim is not None:
        #         shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
        #         shard_size = self.output_sizes[loaded_shard_id] // tp_size

        #         if packed_dim == output_dim:
        #             shard_size //= param.pack_factor
        #             shard_offset //= param.pack_factor
        #             shard_size, shard_offset = adjust_marlin_shard(param, shard_size, shard_offset)

        #         # 使用较少的复制和索引操作
        #         param_data_ = param_data.narrow(output_dim, shard_offset, shard_size) if self.use_llama_nn else param_data

        #         start_idx = tp_rank * shard_size
        #         loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        #         assert param_data_.shape == loaded_weight.shape
        #         param_data_.copy_(loaded_weight)

        #         if loaded_shard_id == 1 and len(param_data.shape) == 2:
        #             param_data = param_data.transpose(0, 1)
        #             param.data = param_data.reshape(param_data.shape[1], -1)
        #     else:
        #         if not getattr(param, "ignore_warning", False):
        #             logger.warning("Loading a weight without `output_dim` attribute in "
        #                         "MergedColumnParallelLinear, assume the weight is "
        #                         "the same for all partitions.")
        ###########################################################################


class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        linear_method: (Maybe quantized) linear method.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size,
                                               self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_heads +
                       2 * self.num_kv_heads) * tp_size * self.head_size
        super().__init__(input_size, output_size, bias, False, skip_bias_add,
                         params_dtype, linear_method)
        self.use_llama_nn = os.environ.get('LLAMA_NN') == '1'

    # @torch.compile #######编译优化
    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[str] = None):
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)

        if loaded_shard_id is None:
            # Loaded weight is already packed.
            if output_dim is None:
                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.total_num_heads * self.head_size),
                ("k", self.total_num_heads * self.head_size,
                 self.total_num_kv_heads * self.head_size),
                ("v", (self.total_num_heads + self.total_num_kv_heads) *
                 self.head_size, self.total_num_kv_heads * self.head_size),
            ]
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor

                    # If marlin, we need to adjust the offset and size to account for the tiling.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset)

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                ######self.weight_loader(param, loaded_weight_shard, shard_id)
                #########################
                ###self.weight_loader(param, loaded_weight_shard, shard_id)
                # 批量加载权重
                param_data.copy_(loaded_weight_shard)  # 直接复制而不是逐个调用
                #########################
            return

        tp_rank = get_tensor_model_parallel_rank()
        assert loaded_shard_id in ["q", "k", "v"]
        if output_dim is not None:
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_heads +
                                self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor

                # If marlin, we need to adjust the offset and size to account for the tiling.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset)

            if self.use_llama_nn:
                param_data_ = param_data.narrow(output_dim, shard_offset,
                                           shard_size)
            else:
                param_data = param_data.narrow(output_dim, shard_offset,
                                           shard_size)
            if loaded_shard_id == "q":
                shard_id = tp_rank
            else:
                shard_id = tp_rank // self.num_kv_head_replicas
            start_idx = shard_id * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions.")
        
        if self.use_llama_nn:
            assert param_data_.shape == loaded_weight.shape
            param_data_.copy_(loaded_weight)
            if loaded_shard_id == "v" and len(param_data.shape) == 2:
                param_data = param_data.transpose(0, 1) 
                param.data = param_data.reshape(param_data.shape[1], -1) 
        else:
            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)


class RowParallelLinear(torch.nn.Module):


    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        linear_method: (Maybe quantized) linear method.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        # Divide the weight matrix along the last dimension.
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.skip_bias_add = skip_bias_add
        if linear_method is None:
            linear_method = UnquantizedLinearMethod()
        self.linear_method = linear_method
        self.linear_weights = self.linear_method.create_weights(
            self.input_size_per_partition, self.output_size, self.input_size,
            self.output_size, self.params_dtype)
        for name, weight in self.linear_weights.items():
            if isinstance(weight, torch.Tensor):
                self.register_parameter(name, weight)
                set_weight_attrs(weight, {"weight_loader": self.weight_loader})

        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)
        self.use_llama_nn = os.environ.get('LLAMA_NN') == '1'

    # @torch.compile #######编译优化
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        input_dim = getattr(param, "input_dim", None)
        param_data = param.data
        if input_dim is not None:
            shard_size = param_data.shape[input_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                 shard_size)
        assert param_data.shape == loaded_weight.shape
        if self.use_llama_nn:
            loaded_weight = loaded_weight.transpose(0, 1)
            loaded_weight=loaded_weight.reshape(param_data.shape[0],-1)
        param_data.copy_(loaded_weight)
    
    # @torch.compile #######编译优化
    def forward(self, input_):
        # Set up backprop all-reduce. ####检查输入是否已经被并行化
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim( ####前输入尚未并行化。为了并行化输入数据，代码通过 split_tensor_along_last_dim 方法，按照最后一个维度（通常是特征维度）将输入张量划分为多个部分，每个 GPU 得到其中的一部分数据
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()####contiguous: 确保选中的输入块是内存上连续的，以便后续操作中可以直接使用，避免潜在的性能问题
        #######################################
        # 确保输入是合适的形状，以便进行矩阵乘法
        input_shape = input_parallel.shape
        if len(input_shape) == 2:  # 如果输入是2D
            input_parallel = input_parallel.unsqueeze(1)  # 添加一个维度以兼容批处理
        #######################################
        # Matrix multiply.
        

        """apply_weights: 这个函数负责执行矩阵乘法，主要是将输入张量与权重矩阵相乘。
        self.linear_method 是线性计算的实现类，可能是未量化或量化的线性方法，允许模型根据不同需求使用不同的实现。
        self.linear_weights 包含了当前层的权重信息，它们是按照并行策略进行存储和划分的。
        在这里，矩阵乘法的计算是并行化的，因为每个 GPU 都只负责输入张量的一部分，计算该部分的输出"""
        
        output_parallel = self.linear_method.apply_weights(
            self.linear_weights, input_parallel)
        if self.reduce_results and self.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias


# class RowParallelLinear(torch.nn.Module):
    """线性层带行并行优化。"""
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 input_is_parallel: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 reduce_results: bool = True,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        self.skip_bias_add = skip_bias_add
        logger.info("算子优化第一步")

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        # 获取并行度信息
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, self.tp_size)

        if linear_method is None:
            linear_method = UnquantizedLinearMethod()
        self.linear_method = linear_method
        self.linear_weights = self.linear_method.create_weights(
            self.input_size_per_partition, self.output_size, self.input_size,
            self.output_size, self.params_dtype)

        for name, weight in self.linear_weights.items():
            if isinstance(weight, torch.Tensor):
                self.register_parameter(name, weight)
                set_weight_attrs(weight, {"weight_loader": self.weight_loader})

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {"output_dim": 0, "weight_loader": self.weight_loader})
        else:
            self.register_parameter("bias", None)
        
        # 是否使用编译优化
        self.use_llama_nn = os.environ.get('LLAMA_NN') == '1'

    @torch.compile  # 编译优化
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        input_dim = getattr(param, "input_dim", None)
        param_data = param.data

        if input_dim is not None:
            shard_size = param_data.shape[input_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
        
        assert param_data.shape == loaded_weight.shape

        if self.use_llama_nn:
            loaded_weight = loaded_weight.transpose(0, 1).reshape(param_data.shape[0], -1)
        
        param_data.copy_(loaded_weight)

    @torch.compile  # 编译优化
    def forward(self, input_):
        # 检查输入是否已经并行化
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()  # 保证内存连续性
        
        # 确保输入形状正确以进行矩阵乘法
        input_parallel = input_parallel if input_parallel.ndim > 2 else input_parallel.unsqueeze(1)
        
        # 并行化的矩阵乘法
        output_parallel = self.linear_method.apply_weights(self.linear_weights, input_parallel)

        if self.reduce_results and self.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        # 处理偏置
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias

        return output, output_bias