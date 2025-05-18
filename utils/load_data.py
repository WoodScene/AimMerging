
from datasets import concatenate_datasets, load_dataset
from utils.dataset_order import get_dataset_order
import os
import sys
import random
from datasets import Dataset
from collections import Counter

def load_current_task_data(dataset_id, task_id, data_dir, cache_dir=None, model_name ="t5"):
    """
    加载当前任务的数据

    参数:
    - task_id (int): 当前任务的 ID，用于指定要加载的文件。
    - data_dir (str): 数据的根目录，存放每个任务的数据文件。
    - cache_dir (str, 可选): 数据缓存目录，可避免重复加载。

    返回:
    - Dataset: 返回 Hugging Face 格式的 Dataset 对象。
    """
    # 构建当前任务数据文件路径
    dataset_order = get_dataset_order(dataset_id)
    
    if "t5" in model_name:
        data_path = os.path.join(data_dir, "train", dataset_order[task_id] + "_T5.json")
    else:
        data_path = os.path.join(data_dir, "train", dataset_order[task_id] + ".json")
    print(f"current data path: {data_path}")
    assert os.path.exists(data_path), "data_path not find!"


    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_path,cache_dir=cache_dir)['train']
    else:
        dataset = load_dataset(data_path,cache_dir=cache_dir)['train']
    print(f"总样本数：{len(dataset)}")

    return dataset

def load_memory_buffer_old(dataset_id, task_id, data_dir, sampling_ratio, cache_dir, model_name ="t5", random_seed: int = 42):
    """
    从历史任务中加载数据作为 memory buffer。

    Args:
        task_id (int): 当前任务的 ID memory buffer 只包括从 task 1 到 task (task_id-1) 的数据。
        sampling_ratio (int): 每个任务数据的采样比例，取值范围在 0 和 100 之间。
        random_seed (int): 随机种子，保证采样的一致性。

    Returns:
        MemoryBufferDataset: 包含采样数据的 memory buffer 数据集。
    """
    random.seed(random_seed)
    dataset_order = get_dataset_order(dataset_id)
    buffer_data = []

    for i in range(task_id):
        # 加载每个历史任务的数据文件
        data_path = os.path.join(data_dir, "train", dataset_order[i] + "_T5.json")
        print(f"task id: {i}, history data path: {data_path}")
        assert os.path.exists(data_path), "data_path not find!"

        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            task_data = load_dataset("json", data_files=data_path,cache_dir=cache_dir)["train"]
        else:
            task_data = load_dataset(data_path,cache_dir=cache_dir)["train"]

        # 根据采样比例进行采样
        num_samples = int(len(task_data) * sampling_ratio / 100)
        sampled_data = task_data.shuffle(seed=random_seed).select(range(num_samples))

        # 将采样数据添加到 memory buffer 中
        buffer_data.append(sampled_data)
        print(f"总样本数：{len(task_data)}, 选择的样本数量：{num_samples}")

    # 将所有采样数据转为 Hugging Face 的 Dataset 格式
    memory_buffer = buffer_data[0]  # 先使用第一个数据集
    for dataset in buffer_data[1:]:
        memory_buffer = concatenate_datasets([memory_buffer, dataset])
    return memory_buffer


# 按照每个类别进行采样
def load_memory_buffer(dataset_id, task_id, data_dir, sampling_ratio, cache_dir, model_name ="t5", random_seed: int = 42):
    """
    从历史任务中加载数据作为 memory buffer。

    Args:
        task_id (int): 当前任务的 ID memory buffer 只包括从 task 1 到 task (task_id-1) 的数据。
        sampling_ratio (int): 每个任务数据的采样比例，取值范围在 0 和 100 之间。
        random_seed (int): 随机种子，保证采样的一致性。

    Returns:
        MemoryBufferDataset: 包含采样数据的 memory buffer 数据集。
    """
    random.seed(random_seed)
    dataset_order = get_dataset_order(dataset_id)
    buffer_data = []

    for i in range(task_id):
        # 加载每个历史任务的数据文件
        if "t5" in model_name:
            data_path = os.path.join(data_dir, "train", dataset_order[i] + "_T5.json")
        else:
            data_path = os.path.join(data_dir, "train", dataset_order[i] + ".json")
        print(f"task id: {i}, history data path: {data_path}")
        assert os.path.exists(data_path), "data_path not find!"

        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            task_data = load_dataset("json", data_files=data_path,cache_dir=cache_dir)["train"]
        else:
            task_data = load_dataset(data_path,cache_dir=cache_dir)["train"]

        # 根据采样比例进行采样
        num_samples = int(len(task_data) * sampling_ratio / 100)


        # 计算类别分布
        label_column = "output"  # 需要根据实际数据集的类别字段名称调整
        label_counts = Counter(task_data[label_column])
        num_classes = len(label_counts)
        samples_per_class = max(1, num_samples // num_classes)

        # 对每个类别进行采样
        sampled_indices = []
        for label in label_counts:
            label_indices = [idx for idx, value in enumerate(task_data[label_column]) if value == label]
            sampled_indices.extend(random.sample(label_indices, min(samples_per_class, len(label_indices))))

        # 如果采样数量不足 num_samples，则从剩余数据中随机补充
        if len(sampled_indices) < num_samples:
            remaining_indices = list(set(range(len(task_data))) - set(sampled_indices))
            sampled_indices.extend(random.sample(remaining_indices, num_samples - len(sampled_indices)))

        # 根据采样的索引选择数据
        sampled_data = task_data.select(sampled_indices)

        #sampled_data = task_data.shuffle(seed=random_seed).select(range(num_samples))

        # 将采样数据添加到 memory buffer 中
        buffer_data.append(sampled_data)
        print(f"总样本数：{len(task_data)}, 选择的样本数量：{num_samples}")

    # 将所有采样数据转为 Hugging Face 的 Dataset 格式
    memory_buffer = buffer_data[0]  # 先使用第一个数据集
    for dataset in buffer_data[1:]:
        memory_buffer = concatenate_datasets([memory_buffer, dataset])
    return memory_buffer




def load_validation_set(data_dir, dataset_id, task_id, cache_dir, model_name ="t5", val_set_size_per_task=25, seed=42):
    """
    从 ./dev 文件夹中加载验证集数据，根据 task_id 加载历史任务数据并根据指定数量采样。

    Args:
        dev_dir (str): 验证集存储的目录路径。
        val_set_size (int): 验证集采样数量。
        task_id (int): 当前任务的 ID，加载历史任务数据。
        seed (int): 随机种子，保证采样的一致性。

    Returns:
        val_data: 验证集数据，Hugging Face Dataset 格式。
    """
    # 设置随机种子
    random.seed(seed)

    # 存储所有的验证集样本
    val_data_list = []
    dataset_order = get_dataset_order(dataset_id)
    # 遍历 ./dev 文件夹中所有 JSON 文件
    for i in range(task_id+1):
        if "t5" in model_name:
            data_path = os.path.join(data_dir, "dev", dataset_order[i] + "_T5.json")
        else:
            data_path = os.path.join(data_dir, "dev", dataset_order[i] + ".json")
        assert os.path.exists(data_path)

        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            task_data = load_dataset("json", data_files=data_path,cache_dir=cache_dir)["train"]
        else:
            task_data = load_dataset(data_path,cache_dir=cache_dir)["train"]



        # 打印原始数据量
        # print(f"Processing validation file: {file_name}, Total samples: {len(data)}")
        #random.shuffle(data)
        # 添加数据到验证集列表中
        #sampled_data = data[:val_set_size_per_task]  # val_set_size_per_task 为每个任务采样的样本数量
        sampled_data = task_data.shuffle(seed=42).select(range(min(len(task_data), val_set_size_per_task)))
        

        val_data_list.append(sampled_data)

    if len(val_data_list) > 1:
        val_data = concatenate_datasets(val_data_list)
    else:
        val_data = val_data_list[0]
    
    return val_data
