# 生成计算fwt指标所需要的预测结果
# 这里只需要计算每一个task 单独微调后的性能就可以了，另外一部分的结果在BWT中计算了
# where a0,t refers to the performance of training task t individually

import os
import sys
import json
import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


from utils.dataset_order import get_dataset_order


def main(
    load_8bit: bool = True,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    testfile_name: str = "",
    testfile_idx: str = "",
    output_file: str = "",
    dataset_id: int = 1, # 1 - 5  5次实验
    max_new_tokens: int = 128, # 1 - 5  5次实验
    service_begin_id: int = 0 ,
    data_dir: str = "./data_longsequence",

):

    print(f"base_model: {base_model}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"dataset_id: {dataset_id}")
    print(f"service_begin_id: {service_begin_id}")
    print(f"data_dir: {data_dir}")
    dataset_order = get_dataset_order(dataset_id)
    service_name = dataset_order[service_begin_id]
    model_name = base_model.split("/")[-1] + "lora"

    print(f"model_name: {model_name}")


    lora_weights = os.path.join("./checkpoint_files", model_name + "_FWT" , str(service_begin_id)+"-"+service_name)
    if not os.path.exists(lora_weights):
        print(f"lora dir {lora_weights} not find!")
        sys.exit(1)   
    assert (
        lora_weights
    ), "Please specify a --lora_weights, e.g. --lora_weights='xxx'"

    output_dir = os.path.join("./output", model_name + "_fwt")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"lora_weights: {lora_weights}")
    print(f"output_dir: {output_dir}")
    
    
    
    prompter = Prompter(prompt_template)
    #tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, device_map="auto")
    #print(torch.cuda.is_available())
    #sys.exit(1)
    if device == "cuda":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model, 
            load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    else:
        print("model cuda error")
        sys.eixt(1)
    #print(model.config.use_cache)
    #sys.exit(1)
    # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)



    service_id = service_begin_id
    print(f"current service name: {dataset_order[service_id]}... begin generating!")
    
    output_file = os.path.join(output_dir, dataset_order[service_id] +"_result.txt")
    print(f"output filename: {output_file}")
    
    #testfile_name = "./data_longsequence/test/" + dataset_order[service_id] + "_T5.json"
    testfile_name = os.path.join(data_dir, "test/" + dataset_order[service_id] + "_T5.json")
        
    print(f"test filename: {testfile_name}")
    

    if not os.path.isfile(output_file): # 如果文件不存在，就新建文件，从头开始写入
        result_out = open(output_file, "w", encoding='utf-8')
        begin_id = 0 # 相当于从头开始获取结果
        print("——————————————————————————————从头开始写入——————————————————————————————")
    else: # 如果文件已经存在了，我们看看已经写了多少行了，需要继续从哪里开始获取结果
        with open(output_file, "r") as f:
            lines = f.readlines()
            begin_id = len(lines)
            f.close()
        print(f"——————————————————————————————从第{begin_id}行开始写入——————————————————————————————")
        result_out = open(output_file, "a", encoding='utf-8')
    
    
    data = json.load(open(testfile_name)) 
    for idx_ in range(begin_id, len(data)):
        sample = data[idx_]

        Response_list = []

        input_ids = tokenizer(sample['input'], return_tensors='pt').input_ids.cuda()
        output = model.generate(input_ids=input_ids, max_length=max_new_tokens)
        answer = tokenizer.decode(output[0])
        if "</s>" in answer:
            answer = answer.replace("</s>","")
        if "<pad>" in answer:
            answer = answer.replace("<pad>","")
                
        Response_list.append(answer.strip())
        #print("Input:", input2)
        print("Input:", sample['input'])
        print("Response list:", Response_list)
        print("Ground truth:", sample['output'])
        print()
        # if "NONE" not in Response:
        #     break
        # if sample['output'] != "NONE":
        #     break
        #result_out.write(idx_line + "|||" + str(Response_list))
        result_out.write(sample['id'] + "|||" + sample['output'] + "|||" + str(Response_list))
            
        result_out.write("\n")

        #break
    result_out.close()
    print(f"current service name: {service_name}... Generate End!")

if __name__ == "__main__":
    fire.Fire(main)
