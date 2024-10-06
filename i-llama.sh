#!/bin/bash
#SBATCH -J test
#SBATCH -p wzidnormal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=dcu:1
#SBATCH -t 1-0
module load apps/anaconda3/5.2.0
source /public/software/apps/anaconda3/5.2.0/bin/activate
conda activate llama333
module rm compiler/dtk/21.10
module load compiler/dtk/24.04
export LOG_LEVEL='DEBUG'
export LD_LIBRARY_PATH="/work/home/xdb4_60320/xdb-www/das1.0/das1_0/rocblas-install/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/work/home/xdb4_60320/xdb-www/das1.0/das1_0/vllm/vllm:$PYTHONPATH"
/work/home/xdb4_60320/.conda/envs/llama333/bin/python3.1 -m pip install --upgrade pip
pip install -e /work/home/xdb4_60320/xdb-www/das1.0/das1_0/vllm/ 
# python /work/home/xdb4_60320/xdb-www/das1.0/das1_0/vllm/setup.py install
HIP_VISIBLE_DEVICES=0 python /work/home/xdb4_60320/xdb-www/das1.0/das1_0/vllm/benchmarks/benchmark_throughput.py --num-prompts 32 --model /work/models/Meta-Llama-3-8B --dataset /work/models/data/ShareGPT_V3_unfiltered_cleaned_split.json --trust-remote-code  --enforce-eager --dtype float16
