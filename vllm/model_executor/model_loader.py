import contextlib
from typing import Type
import torch
import torch.nn as nn
from vllm.config import DeviceConfig, ModelConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.weight_utils import get_quant_config, initialize_dummy_weights
import os

# 缓存环境变量和 GPU 能力，避免重复查询
LLAMA_NN = os.getenv('LLAMA_NN')
CAPABILITY = None

if torch.cuda.is_available():
    CAPABILITY = torch.cuda.get_device_capability()
    CAPABILITY = CAPABILITY[0] * 10 + CAPABILITY[1]


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(model_config: ModelConfig) -> Type[nn.Module]:
    architectures = getattr(model_config.hf_config, "architectures", [])
    
    # 只在必要时设置环境变量
    global LLAMA_NN
    if architectures in [['LlamaForCausalLM'], ['ChatGLMModel'], ['BaichuanForCausalLM']]:
        if LLAMA_NN != '0': 
            os.environ['LLAMA_NN'] = '1'
            LLAMA_NN = '1'
    
    # 处理 Mixtral 的特殊情况
    if model_config.quantization and "MixtralForCausalLM" in architectures:
        architectures = ["QuantMixtralForCausalLM"]

    # 使用 ModelRegistry 选择正确的模型架构
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls

    # 如果没有匹配的架构，抛出异常
    raise ValueError(
        f"Model architectures {architectures} are not supported. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_model(model_config: ModelConfig, device_config: DeviceConfig, **kwargs) -> nn.Module:
    lora_config = kwargs.get("lora_config", None)
    model_class = _get_model_architecture(model_config)

    # 获取线性方法（可能是量化）
    linear_method = None
    if model_config.quantization:
        quant_config = get_quant_config(model_config)
        
        # 利用之前缓存的 GPU 能力
        if CAPABILITY < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} is not "
                "supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {CAPABILITY}."
            )
        
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}")
        linear_method = quant_config.get_linear_method()

    # 检查是否需要禁用 LLAMA_NN
    if linear_method is not None and LLAMA_NN != '0':
        os.environ['LLAMA_NN'] = '0'
        LLAMA_NN = '0'

    with _set_default_torch_dtype(model_config.dtype):
        # 创建模型实例
        with torch.device(device_config.device):
            if hasattr(model_class, "supported_lora_modules"):
                model = model_class(model_config.hf_config, linear_method, lora_config)
            elif lora_config:
                raise ValueError(
                    f"Model {model_class.__name__} does not support LoRA, "
                    "but LoRA is enabled. Support for this model may "
                    "be added in the future. If this is important to you, "
                    "please open an issue on github."
                )
            else:
                model = model_class(model_config.hf_config, linear_method)

        # 根据加载格式加载权重
        if model_config.load_format == "dummy":
            initialize_dummy_weights(model)
        else:
            model.load_weights(model_config.model, model_config.download_dir,
                               model_config.load_format, model_config.revision)
    
    return model.eval()
