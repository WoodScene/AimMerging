# 新的框架

import os
import matplotlib.pyplot as plt

import sys
from typing import List
from peft import PeftModel
import fire
import torch
import transformers
import shutil
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import gc
import numpy as np
import pandas as pd

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
import time
from utils.lora_importance_bilevel_adaptive import RankAllocator
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter
from utils.prompter import Prompter
from transformers import set_seed
from utils.dataset_order import get_dataset_order

set_seed(42)
from utils.load_data import load_current_task_data, load_memory_buffer, load_validation_set

def train(
    # 可以调节的超参数
    method_name: str = "",  # the only required argument
    inner_iterations: int = 4, #new
    outer_iterations: int = 4, #new
    adaptive_merge_flag: bool = False,
    cal_vanilla_para_change_flag: bool = False,
    cold_start_step: int = 3, #new
    max_step_len: int = 128, #new
    min_step_len: int = 2, #new
    train_batch_size_outer: int = 8,#new 16
    empty_inner_score_flag: int = 0, #是否清空累积的重要性梯度，0表示不清空 new 0 1
    empty_outer_score_flag: int = 0, #new 0 1
    quantile: float = 0.8,
    threshold_factor: float = 2.0,
    # model/data params
    base_model: str = "",  # the only required argument
    data_dir: str = "./data_longsequence",
    output_path: str = "./checkpoint_files",
    cache_dir: str = "/",
    # training hyperparams
    batch_size: int = 8, # 16
    micro_batch_size: int = 2, # 4
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    outer_lr: float = 1e-3, #new
    max_input_length: int = 1024,
    max_target_length: int = 128,
    val_set_size: int = 100,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q", "v"
    ],
    # llm hyperparams
    ignore_pad_token_for_loss: bool = True,
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = True,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    # CL hyperparams
    dataset_id: int = 1, # 1 - 5  5次实验
    task_id: int = 0, # 这个表示从哪个service开始训练，默认从头开始训练
    beta1: float = 0.85, 
    beta2: float = 0.85,
    memory_data_ratio: int = 2, # 这个表示训练数据的比例；
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training T5 model with params:\n"
            f"base_model: {base_model}\n"
            f"method_name: {method_name}\n"
            f"batch_size: {batch_size}\n"
            f"quantile: {quantile}\n"
            f"beta1: {beta1}\n"
            f"beta2: {beta2}\n"
            f"train_batch_size_outer: {train_batch_size_outer}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"inner_iterations: {inner_iterations}\n"
            f"outer_iterations: {outer_iterations}\n"
            f"cold_start_step: {cold_start_step}\n"
            f"max_step_len: {max_step_len}\n"
            f"min_step_len: {min_step_len}\n"
            f"adaptive_merge_flag: {adaptive_merge_flag}\n"
            f"threshold_factor: {threshold_factor}\n"
            f"cal_vanilla_para_change_flag: {cal_vanilla_para_change_flag}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"max_input_length: {max_input_length}\n"
            f"max_target_length: {max_target_length}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"ignore_pad_token_for_loss: {ignore_pad_token_for_loss}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    assert (
        method_name
    ), "Please specify a --method_name, e.g. --method_name='bilevel'"
    
    dataset_order = get_dataset_order(dataset_id)
    
    # new  新添加内容
    # 遍历每一个service
    # 注意下一个要从上一个的checkpoint处继续开始
    # 支持断点回复操作
    print(f"current service name: {dataset_order[task_id]}... begin fine tuning!")
    print(f"memory_data_ratio is : {memory_data_ratio/100}")
    print(f"data_dir is : {data_dir}")
    model_name = base_model.split("/")[-1] + "lora"
    output_dir = os.path.join(output_path, model_name + "_"+ method_name +"_dataset_id_"+str(dataset_id), str(task_id)+"-"+dataset_order[task_id])
    
    # print(f"output_dir: {output_dir}")
    # if os.path.exists(output_dir):
    #     print(f"output_dir dir {output_dir} exist!")
    #     sys.exit(1)
    

    # 首先需要检查一下上一个service的checkpoint文件是否存在
    if task_id == 0:
        lora_weights = ""
    else:
        last_service_name = dataset_order[task_id - 1]
        last_checkpoint_dir = os.path.join(output_path, model_name + "_"+ method_name +"_dataset_id_"+str(dataset_id), str(task_id-1)+"-"+last_service_name)
        lora_weights = last_checkpoint_dir
        if not os.path.exists(lora_weights):
            print(f"lora_weights dir {lora_weights} not find!")
            sys.exit(1)
    print(f"lora_weights: {lora_weights}\n")

    # 获取当前数据和memory buffer
    data = load_current_task_data(dataset_id, task_id, data_dir, cache_dir)
    if task_id > 0: # task_id是从0开始
        memory_data = load_memory_buffer(dataset_id, task_id, data_dir, memory_data_ratio, cache_dir)
    else:
        memory_data = None

    # print(data)
    # print()
    # print(memory_data)

    # Dataset({
    #     features: ['id', 'input', 'output'],
    #     num_rows: 1073
    # })

    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("DDP 启动！")

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model, 
        # torch_dtype=torch.bfloat16, 
        device_map=device_map,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, device_map="auto")
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1,2,none")

    padding = False # "max_length"
    prefix = ""
    def preprocess_function(examples):
        inputs = examples['input'] 
        targets = examples['output']
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=max_input_length, padding=padding, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    if task_id == 0:
        model = get_peft_model(model, config)
        print("fine tune lora from scratch!")
    # https://github.com/tloen/alpaca-lora/issues/44
    else:
        model = PeftModel.from_pretrained(model, lora_weights, is_trainable=True)
        print("continual fine tune lora!")

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    
    if val_set_size > 0:

        val_data = load_validation_set(data_dir, dataset_id, task_id, cache_dir)

        val_data = (
            val_data.shuffle(seed=42).map(preprocess_function, remove_columns=['input', 'output'], batched=True)
        )

        train_data = (
            data.shuffle().map(preprocess_function,remove_columns=['input', 'output'], batched=True)
        )
        # 按比例来设置val_set_size吧
        # val_set_len = int(len(data) * 10 / 100)
        # train_val = data.train_test_split(
        #     test_size=val_set_len, shuffle=True, seed=42
        # )
        # train_data = (
        #     train_val["train"].shuffle().map(preprocess_function,remove_columns=['input', 'output'], batched=True)
        # )
        # val_data = (
        #     train_val["test"].shuffle().map(preprocess_function,remove_columns=['input', 'output'], batched=True)
        # )
        print(f"训练数据总量：{len(train_data)}")
        print(f"验证数据总量：{len(val_data)}")
    else:
        train_data = data.shuffle().map(preprocess_function,remove_columns=['input', 'output'], batched=True)
        val_data = None
    

    # new 记得添加这两行
    if memory_data is not None:
        train_data_memory = memory_data.shuffle().map(preprocess_function,remove_columns=['input', 'output'], batched=True)
        print(f"memory数据总量：{len(train_data_memory)}")
    else:
        train_data_memory = None

 


    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    rankallocator = RankAllocator(
        model,
        init_warmup=50,
        beta1=beta1, 
        beta2=beta2, 
        quantile=quantile,
        rank=lora_r,
        taylor="param_first",
    )


    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        train_dataset_memory=train_data_memory, # new
        eval_dataset=val_data,
        ipt_score = rankallocator,
        outer_lr=outer_lr, # new
        empty_inner_score_flag=empty_inner_score_flag, # new
        empty_outer_score_flag=empty_outer_score_flag, # new
        outer_iterations=outer_iterations, # new
        adaptive_merge_flag=adaptive_merge_flag,
        cal_vanilla_para_change_flag=cal_vanilla_para_change_flag,
        cold_start_step = cold_start_step, #new
        max_step_len = max_step_len, #new
        min_step_len = min_step_len, #new
        threshold_factor = threshold_factor,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            train_batch_size_outer=train_batch_size_outer,# new
            inner_iterations=inner_iterations, # new
            warmup_steps=50,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=40 if val_set_size > 0 else None,
            save_steps=40,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if train_data_memory:
        delta_in_abs_change_list, delta_in_abs_change_step_list, inner_step_list = rankallocator.show_delta_in_abs_change()
        print(f"delta_in_abs_change_list: {delta_in_abs_change_list}")
        print(f"delta_in_abs_change_step_list: {delta_in_abs_change_step_list}")
                
        output_dir2 = os.path.join("./Fig", "parameter_change_plot_" + model_name + "_"+ method_name +"_dataset_id_"+str(dataset_id))
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)
        # 创建一个新的图表
        if delta_in_abs_change_step_list and delta_in_abs_change_list and inner_step_list:
            plt.figure(figsize=(10, 6))
            plt.plot(delta_in_abs_change_step_list, delta_in_abs_change_list, marker='o', linestyle='-', color='b', label='Parameter Change')
            plt.title('Parameter Change Over Training Steps')
            plt.xlabel('Training Step')
            plt.ylabel('Absolute Change in Parameters')
            plt.grid(True)
            plt.legend()
            fig_save_name = str(task_id)+"-"+dataset_order[task_id] + ".png"
            plt.savefig(os.path.join(output_dir2, fig_save_name), dpi=300)  # 保存为 PNG 格式，分辨率为 300 dpi
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(delta_in_abs_change_step_list, inner_step_list, marker='o', linestyle='-', color='b', label='Parameter Change')
            plt.title('Inner step Over Training Steps')
            plt.xlabel('Training Step')
            plt.ylabel('Inner Step')
            plt.grid(True)
            plt.legend()
            fig_save_name = str(task_id)+"-"+dataset_order[task_id] + "_step.png"
            plt.savefig(os.path.join(output_dir2, fig_save_name), dpi=300)  # 保存为 PNG 格式，分辨率为 300 dpi
            plt.close()
        else:
            print("Warning: Lists are empty. Skipping plot.")

        # 再查看一下history and train loss的变化情况
        history_loss_list, history_loss_step_list, tr_loss_list = rankallocator.show_history_loss()
        plt.figure(figsize=(10, 6))
        plt.plot(history_loss_step_list, history_loss_list, marker='o', linestyle='-', color='b', label='History Loss Change')
        plt.title('History Loss Change Over Training Steps')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        fig_save_name = str(task_id)+"-"+dataset_order[task_id] + "_historyloss.png"
        plt.savefig(os.path.join(output_dir2, fig_save_name), dpi=300)  # 保存为 PNG 格式，分辨率为 300 dpi
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(history_loss_step_list, tr_loss_list, marker='o', linestyle='-', color='b', label='Train Loss Change')
        plt.title('Training Loss Change Over Training Steps')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        fig_save_name = str(task_id)+"-"+dataset_order[task_id] + "_trainloss.png"
        plt.savefig(os.path.join(output_dir2, fig_save_name), dpi=300)  # 保存为 PNG 格式，分辨率为 300 dpi
        plt.close()

        # 将三个列表转换为 Pandas DataFrame
        df = pd.DataFrame({
            'Step': delta_in_abs_change_step_list,
            'Delta Change': delta_in_abs_change_list,
            'Inner Step': inner_step_list
        })
        # 将 DataFrame 保存为 CSV 文件
        output_dir2 = os.path.join("./csv_file", model_name + "_"+ method_name +"_dataset_id_"+str(dataset_id))
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)
        csv_save_name = str(task_id)+"-"+dataset_order[task_id] + ".csv"
        
        df.to_csv(os.path.join(output_dir2, csv_save_name), index=False)

        df2 = pd.DataFrame({
            'Step': history_loss_step_list,
            'History Loss': history_loss_list,
            'Train Loss': tr_loss_list, 
        })
        csv_save_name = str(task_id)+"-"+dataset_order[task_id] + "_loss.csv"
        
        df2.to_csv(os.path.join(output_dir2, csv_save_name), index=False)

    if cal_vanilla_para_change_flag:
        vanilla_abs_change_list, vanilla_abs_change_step_list = rankallocator.get_delta_vanilla_abs_change()
        plt.figure(figsize=(10, 6))
        plt.plot(vanilla_abs_change_step_list, vanilla_abs_change_list, marker='o', linestyle='-', color='b', label='Parameter Change')
        plt.title('Parameter Change Over Training Steps')
        plt.xlabel('Training Step')
        plt.ylabel('Absolute Change in Parameters')
        plt.grid(True)
        plt.legend()

        output_dir2 = os.path.join("./Fig", "parameter_change_plot_" + model_name + "_"+ method_name +"_dataset_id_"+str(dataset_id))
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)
        fig_save_name = str(task_id)+"-"+dataset_order[task_id] + "_vanilla_train.png"
        plt.savefig(os.path.join(output_dir2, fig_save_name), dpi=300)  # 保存为 PNG 格式，分辨率为 300 dpi
        plt.close()
        
        output_dir2 = os.path.join("./csv_file", model_name + "_"+ method_name +"_dataset_id_"+str(dataset_id))
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)
        df2 = pd.DataFrame({
            'Step': vanilla_abs_change_step_list,
            'Delta Change': vanilla_abs_change_list,
        })
        csv_save_name = str(task_id)+"-"+dataset_order[task_id] + "_vanilla_train.csv"
        
        df2.to_csv(os.path.join(output_dir2, csv_save_name), index=False)




    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

    


if __name__ == "__main__":
    fire.Fire(train)
