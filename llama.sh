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
HIP_VISIBLE_DEVICES=0 python /work/home/xdb4_60320/xdb-www/das1.0/das1_0/vllm/benchmarks/benchmark_throughput.py --enforce-eager -tp 1 --num-prompts 8 --model /work/models/Meta-Llama-3-8B/ --dtype float16 --trust-remote-code --input-len 8 --output-len 8
