import os
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.logger import init_logger
logger = init_logger(__name__)


KVCache = Tuple[torch.Tensor, torch.Tensor]

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig, linear_method=None) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = None
        self.sampler = Sampler(config.vocab_size)
        self.seq_ids_cache = None  # 预分配 seq_ids 缓存
        logger.info("some thing happen here")

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                kv_caches: List[KVCache], input_metadata: InputMetadata) -> torch.Tensor:
        # 使用 torch.no_grad() 优化推理性能，避免梯度计算
        with torch.no_grad():  
            block_size = self.model.context_buckets[-1]
            if input_metadata.is_prompt:
                if self.seq_ids_cache is None:
                    # 仅在必要时计算 seq_ids 并缓存
                    self.seq_ids_cache = input_metadata.slot_mapping[:, 0] // block_size
                seq_ids = self.seq_ids_cache
            else:
                seq_ids = input_metadata.block_tables

            # 优化后的模型调用方式
            logits = self.model(input_ids, cache_ids=positions, start_ids=seq_ids.flatten())
        return logits

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.model.chkpt_model.lm_head, hidden_states, sampling_metadata)
        return next_tokens

    def load_weights(self, model_name_or_path: str, cache_dir: Optional[str] = None, load_format: str = "auto",
                     revision: Optional[str] = None, **kwargs):
        from transformers_neuronx.llama.model import LlamaForSampling

        # 缓存模型加载路径，避免重复 I/O
        split_model_dir = f"{model_name_or_path}-split"
        model_weights_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        
        if os.path.isdir(model_weights_path):
            split_model_dir = model_name_or_path
        elif not os.path.exists(f"{model_name_or_path}-split"):
            from transformers.models.llama import LlamaForCausalLM
            from transformers_neuronx.module import save_pretrained_split

            hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True)
            save_pretrained_split(hf_model, f"{model_name_or_path}-split")

        # 仅加载一次模型到 Neuron 上，避免重复加载
        if self.model is None:
            self.model = LlamaForSampling.from_pretrained(split_model_dir, **kwargs)
            self.model.to_neuron()
