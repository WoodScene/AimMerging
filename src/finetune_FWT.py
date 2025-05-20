# 单独用每个task 训练一下模型之后查看效果，需要用来计算FWT指标的结果


import os
import sys
from typing import List
from peft import PeftModel
import fire
import torch
import transformers
import shutil
from datasets import load_dataset, concatenate_datasets
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
from utils.load_data import load_current_task_data, load_memory_buffer

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_dir: str = "./data_longsequence",
    output_path: str = "./checkpoint_files",
    cache_dir: str = "/cache",
    # training hyperparams
    batch_size: int = 64, # 64
    micro_batch_size: int = 16, # 16
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
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
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training T5 model with params:\n"
            f"base_model: {base_model}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
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
    
    dataset_order = get_dataset_order(dataset_id)
    
    # new  新添加内容
    # 遍历每一个service
    # 注意下一个要从上一个的checkpoint处继续开始
    # 支持断点回复操作
    print(f"current service name: {dataset_order[task_id]}... begin fine tuning!")
    model_name = base_model.split("/")[-1] + "lora"
    output_dir = os.path.join(output_path, model_name + "_FWT", str(task_id)+"-"+dataset_order[task_id])
    print(f"output_dir: {output_dir}")
    
    # 获取当前数据和memory buffer
    # vanilla training 只需简单合并两个数据集去训练即可
    current_data = load_current_task_data(dataset_id, task_id, data_dir, cache_dir)

    memory_data = None
    data = current_data

    # print(current_data)
    # print()
    # print(memory_data)
    #print(combined_dataset)
    # Dataset({
    #     features: ['id', 'input', 'output'],
    #     num_rows: 1073
    # })


    lora_weights = ""

    print(f"lora_weights: {lora_weights}\n")

    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

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

    model = get_peft_model(model, config)
    print("fine tune lora from scratch!")
    
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
        # 按比例来设置val_set_size吧
        val_set_len = int(len(data) * 10 / 100)
        
        train_val = data.train_test_split(
            test_size=val_set_len, shuffle=True, seed=42
        )
        print(f"训练数据总量：{len(train_val['train'])}")
        print(f"验证数据总量：{len(train_val['test'])}")
        
        train_data = (
            train_val["train"].shuffle().map(preprocess_function,remove_columns=['input', 'output'], batched=True)
        )
        val_data = (
            train_val["test"].shuffle().map(preprocess_function,remove_columns=['input', 'output'], batched=True)
        )
    else:
        train_data = data.shuffle().map(preprocess_function)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
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

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

    


if __name__ == "__main__":
    fire.Fire(train)