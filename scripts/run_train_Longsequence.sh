#!/bin/bash

# 设置起始变量

CUDA_DEVICE=3

begin_id=0
# 初始的迭代步长，后续会自适应调整
inner_step=8
outer_step=$((inner_step))

# 冷启动的步数
cold_start_step=3
# 最长和最短的融合间隔
max_step_len=128
min_step_len=2
# adaptive_merge_flag变成true


method_name="Ours_inner_step_${inner_step}_outer_step_${outer_step}_coldstartstep_${cold_start_step}"

data_id=1

batch_size=32
train_batch_size_outer=32
micro_batch_size=8
# 循环从 begin_id 到 15
for ((ORDER=$begin_id; ORDER<15; ORDER++))
do
    # 执行 Python 文件，传递参数 $i

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python finetune_ours_t5lora.py \
        --base_model '/group/40059/yuujiefeng/Backbones/t5large' \
        --method_name "${method_name}" \
        --num_epochs=10 \
        --dataset_id=${data_id} \
        --task_id=${ORDER} \
        --inner_iterations=${inner_step} \
        --batch_size=${batch_size} \
        --outer_iterations=${outer_step} \
        --train_batch_size_outer=${train_batch_size_outer} \
        --micro_batch_size=${micro_batch_size} \
        --cold_start_step=${cold_start_step} \
        --max_step_len=${max_step_len} \
        --min_step_len=${min_step_len} \
        --empty_inner_score_flag=1 \
        --adaptive_merge_flag='True' \
        --threshold_factor=2.0 \


done


wait


CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python generate_avgPerf_t5lora.py \
    --base_model '/group/40059/yuujiefeng/Backbones/t5large' \
    --dataset_id=${data_id} \
    --method_name "${method_name}" \

wait

# 循环从 begin_id 到 15
for ((ORDER=$begin_id; ORDER<15; ORDER++))
do
    # 执行 Python 文件，传递参数 $i
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python generate_bwt_t5lora.py \
        --base_model '/group/40059/yuujiefeng/Backbones/t5large' \
        --dataset_id=${data_id} \
        --service_begin_id=${ORDER} \
        --method_name "${method_name}" \
        
    # 可以在这里添加任何你需要的其他操作，如等待一段时间等
done
