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
export LD_LIBRARY_PATH=/work/home/xdb4_60320/xdb-www/das1.0/das1_0/rocblas-install/lib:$LD_LIBRARY_PATH
 python -m cProfile -o outputfp6.prof  /work/home/xdb4_60320/xdb-www/das1.0/das1_0/vllm/benchmark_throughput.py --num-prompts 32 --model /work/models/Meta-Llama-3-8B --dataset /work/models/data/ShareGPT_V3_unfiltered_cleaned_split.json --trust-remote-code --enforce-eager --dtype float16
