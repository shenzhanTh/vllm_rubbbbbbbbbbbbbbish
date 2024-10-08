from typing import List, Optional, Union
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.lora.request import LoRARequest
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
from vllm.logger import init_logger
logger = init_logger(__name__)

class LLM:
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        # Default argument optimization
        kwargs.setdefault("disable_log_stats", True)
        # logger.info("使用 kwargs.setdefault() 优化默认参数设置，减少了对 if-else 检查的依赖，提升了代码的可读性和执行效率。")
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        self.request_counter = Counter()

    def get_tokenizer(
        self,
    ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer.tokenizer = tokenizer

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        prefix_pos: Optional[Union[int, List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts."""
        # Pre-validate inputs to reduce checks during execution
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be provided.")
        
        if isinstance(prompts, str):
            prompts = [prompts]  # Convert single prompt to list

        if prompts is not None and prompt_token_ids is not None and len(prompts) != len(prompt_token_ids):
            raise ValueError("Lengths of prompts and prompt_token_ids must match.")

        # Default sampling params optimization
        sampling_params = sampling_params or SamplingParams()

        num_requests = len(prompts or prompt_token_ids)

        # Process each request
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            prefix_pos_i = prefix_pos[i] if prefix_pos is not None else None
            token_ids = prompt_token_ids[i] if prompt_token_ids else None
            self._add_request(prompt, sampling_params, token_ids, lora_request=lora_request, prefix_pos=prefix_pos_i)

        # Execute engine in batch
        return self._run_engine(use_tqdm)

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
        lora_request: Optional[LoRARequest] = None,
        prefix_pos: Optional[int] = None,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id, prompt, sampling_params, prompt_token_ids, lora_request=lora_request, prefix_pos=prefix_pos)

    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        """Run the LLM engine to process requests in batches."""
        outputs = []
        pbar = None
        logger.info("some thing happened in llm.py")
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")

        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)

        if use_tqdm:
            pbar.close()

        # Sort outputs by request_id to ensure correct ordering
        outputs.sort(key=lambda x: int(x.request_id))
        return outputs
