# 按照o-lora中的数据处理格式，生成了prompt模板
# uie_dataset_lora.py line 283

import os
import json
import random
import sys
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
# 15 tasks
# 训练集1000条，验证集500条
task_list = [
    "yelp",
    "amazon",
    "dbpedia",
    "yahoo",
    "agnews",
    "MNLI",
    "QQP",
    "RTE",
    "SST-2",
    "WiC",
    "CB",
    "COPA",
    "BoolQA",
    "MultiRC",
    "IMDB"
]

task_type_dic = {
    "yelp":"SC",
    "amazon":"SC",
    "dbpedia":"TC",
    "yahoo":"TC",
    "agnews":"TC",
    "mnli":"NLI",
    "qqp":"QQP",
    "rte":"NLI",
    "sst-2":"SC",
    "wic":"WiC",
    "cb":"NLI",
    "copa":"BoolQA",
    "boolqa":"BoolQA",
    "multirc":"MultiRC",
    "imdb":"SC"
}

instruction_dic = {
    "NLI": "What is the logical relationship between the \"sentence 1\" and the \"sentence 2\"? Choose one from the option. ",
    "QQP": "Whether the \"first sentence\" and the \"second sentence\" have the same meaning? Choose one from the option. ",
    "SC": "What is the sentiment of the following paragraph? Choose one from the option. ",
    "TC": "What is the topic of the following paragraph? Choose one from the option. ",
    "BoolQA": "According to the following passage, is the question true or false? Choose one from the option. ",
    "MultiRC": "According to the following passage and question, is the candidate answer true or false? Choose one from the option. ",
    "WiC": "Given a word and two sentences, whether the word is used with the same sense in both sentence? Choose one from the option. "
}



# 配置文件夹路径
tasks_folder = './CL_Benchmark/Long_Sequence'
output_folder = './data_longsequence_llama'
train_folder = os.path.join(output_folder, 'train')
val_folder = os.path.join(output_folder, 'dev')
test_folder = os.path.join(output_folder, 'test')

# 确保输出文件夹存在
for folder in [train_folder, val_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)


# 每个任务随机选择样本并保存
for task_name in task_list:
    print(f"task name is {task_name}")
    task_name = task_name.lower()
    task_folder = os.path.join(tasks_folder, task_name)
    
    task_type = task_type_dic[task_name]
    instruction_template = instruction_dic[task_type]
    # print(instruction_template)
    
    label_file = os.path.join(task_folder, 'labels.json')
    with open(label_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    #print(label_data)
    labels_str = ', '.join(label_data)

    instruction_template += "Option: " + labels_str + ".\n" # value of "sentence" will be filled in {0}
    

     
    for data_type in ['train','test','dev']:
        task_file = os.path.join(task_folder, f'{data_type}.json')
        print(f"task_file: {task_file}")
        data_list = []
        # 检查文件是否存在
        if not os.path.exists(task_file):
            print(f'文件 {task_name}.json 不存在，跳过此任务')
            sys.exit(1)

        # 加载任务数据
        with open(task_file, 'r', encoding='utf-8') as f:
            task_data = json.load(f)['Instances']
            
        print(f"{data_type} 数据量为：{len(task_data)}")
        
        random.shuffle(task_data)
        if data_type == "train":
            data = task_data[:1000]
        elif data_type == "dev":
            data = task_data[:500]
        else:
            data = task_data

        id1 = 0
        for tt in data:
            data_item = {}
            data_item['id'] = task_name + "_" + str(id1)
            data_item['instruction'] = instruction_template
            data_item['input'] = tt['input'] + "\nAnswer:" 
            data_item['output'] = tt['output']
            data_list.append(data_item)
    
            id1 += 1
            
        output_folder2 = os.path.join(output_folder, data_type)
        with open(os.path.join(output_folder2, f'{task_name.lower()}.json'), 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)
   

