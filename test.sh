#!/bin/bash
#SBATCH -J test
#SBATCH -p wzidnormal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=dcu:1
#SBATCH -t 1-0
salloc -N 1 -n 8 --gres=dcu:1 -J tess -t 1-0 -p wzidnormal
ssh xdb2
conda activate llama333
module rm compiler/dtk/21.10 
module load compiler/dtk/24.04
export LD_LIBRARY_PATH=/work/home/xdb4_60320/xdb-www/das1.0/das1_0/rocblas-install/lib:$LD_LIBRARY_PATH


python /work/home/xdb4_60320/xdb-www/das1.0/das1_0/vllm/setup.py install
python /work/home/xdb4_60320/xdb-www/das1.0/das1_0/vllm/benchmarks/benchmark_throughput.py --num-prompts 32 --model /work/models/Meta-Llama-3-8B --dataset /work/models/data/ShareGPT_V3_unfiltered_cleaned_split.json --trust-remote-code  --enforce-eager --dtype float16