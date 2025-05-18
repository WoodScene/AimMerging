#!/bin/bash

# FWT指标的计算需要单独训练一下每一个task

# 设置起始变量
begin_id=0

# 最后一个task 需要调小bs，从64\16 to 32\8 

for data_id in 1
do
    # 循环从 begin_id 到 15
    for ((ORDER=$begin_id; ORDER<15; ORDER++))
    do
        # 执行 Python 文件，传递参数 $i

        CUDA_VISIBLE_DEVICES=7 python finetune_FWT_t5lora.py \
            --base_model '/group/40059/yuujiefeng/Backbones/t5large' \
            --num_epochs=10 \
            --dataset_id=${data_id} \
            --task_id=${ORDER} \

    done
done

wait

begin_id=0

for data_id in 1
do
    # 循环从 begin_id 到 15
    for ((ORDER=$begin_id; ORDER<1; ORDER++))
    do
        # 执行 Python 文件，传递参数 $i
        CUDA_VISIBLE_DEVICES=0 python generate_fwt_t5lora.py \
            --base_model '/group/40059/yuujiefeng/Backbones/t5large' \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDER} \

            
        # 可以在这里添加任何你需要的其他操作，如等待一段时间等
    done
done

