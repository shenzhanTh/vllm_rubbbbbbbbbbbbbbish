from mmengine.config import read_base

with read_base():
    from ..datasets.ARC_c.ARC_c_gen_1e0de5 import ARC_c_datasets 
    from ..datasets.ARC_e.ARC_e_gen_1e0de5 import ARC_e_datasets
    from ..summarizers.example import summarizer

datasets = sum([v for k, v in locals().items() if k.endswith("_datasets") or k == 'datasets'], [])
work_dir = './outputs/llama3/'

from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='llama-3-8b-vllm',
        path="/models/llama3/Meta-Llama-3-8B",
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=1,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

