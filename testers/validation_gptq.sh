#!/bin/bash
set -e
source /opt/dtk/env.sh
#[TODO] add vllm build and install
# 解压路径下的dist文件夹
TARGET_DIR="/coursegrader/submit/dist"  
  
# 检查dist目录是否存在  
if [ ! -d "$TARGET_DIR" ]; then  
    echo "错误：'$TARGET_DIR' 目录不存在。"  
    exit 1  
else   
    cd "$TARGET_DIR"  
    echo "安装vllm: $TARGET_DIR"  
    pip install vllm*.whl
fi

cd /opt/opencompass
output=$(python /opt/opencompass/run.py configs/vllm/eval_llama3_gptq_vllm.py)

# echo $output

# output="06/14 10:08:43 - OpenCompass - INFO - write csv to /opt/opencompass/outputs/llama3/20240614_100611/summary/summary_20240614_100611.csv"
csv_path=$(echo $output|awk '{print $NF}') 
echo $csv_path
arc_c=$(awk 'NR==2{split($0, a, ","); print a[length(a)]}' $csv_path)
arc_e=$(awk 'NR==3{split($0, a, ","); print a[length(a)]}' $csv_path)
echo "arc_c:$arc_c arc_e:$arc_e"

if [[ -n "$arc_c" && -n "$arc_e" ]]; then  
    measured_precision=$(echo "($arc_c + $arc_e) / 2" | bc -l)  
    echo "The average accuracy is: $measured_precision"  
else  
    echo "Failed to get both accuracy values."  
fi

expected_precision=81.275 
 
abs_error=$(echo "scale=5; if ($measured_precision < $expected_precision) $expected_precision - $measured_precision else $measured_precision - $expected_precision" | bc)
threshold=4.06375  


if [ $(echo "$measured_precision >= $expected_precision" | bc -l) -eq 1 ]; then
    precision_score=100
elif [ $(echo "$abs_error * 20 < 100" | bc -l) -eq 1 ]; then
    precision_score=$(echo "scale=2; 100-($abs_error * 20)" | bc -l)
else 
    precision_score=0
fi

# echo "abs error:$abs_error precision_score:$precision_score"

echo "precision_score: $precision_score"

#TODO need change dir
# 解压路径下的benchmarks文件夹
cd /coursegrader/submit/benchmarks
generate_throughput_output=$(python benchmark_throughput.py -q gptq --num-prompts 32 --model  /models/llama3/Meta-Llama-3-8B-GPTQ --dataset /models/llama3/data/ShareGPT_V3_unfiltered_cleaned_split.json --trust-remote-code  --enforce-eager --dtype float16 | tail -n 1 )
# generate_throughput_output="Latency: 35.89 s All Throughput: 0.89 requests/s, 366.50 tokens/s Generate Throughput: 189.62 tokens/s"
# echo "generate_throughput_raw: $generate_throughput_output"
generate_throughput=$(echo "$generate_throughput_output" | awk '{print $(NF-1)}')
echo "generate_throughput: $generate_throughput"

min_score_threshold=175   #220
max_score_threshold=275   #320
expected_throughput=225   #270
  
if [ $(echo "$generate_throughput >= $max_score_threshold" | bc -l) -eq 1 ]; then  
    speed_score=100  
elif [ $(echo "$generate_throughput < $min_score_threshold" | bc -l) -eq 1 ]; then  
    speed_score=0  
else  
    speed_score=$(echo "scale=2; ($generate_throughput - $min_score_threshold) * 100 / ($max_score_threshold - $min_score_threshold)" | bc -l )
fi 

echo "speed_score:$speed_score"

final_score=$(echo "$precision_score * 0.2 + $speed_score * 0.8" |bc -l )
echo "final_score:$final_score"