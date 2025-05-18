# 计算JGA指标

import os
import sys
import json
from glob import glob
import argparse
import pandas as pd
from utils.dataset_order import get_dataset_order
from sklearn.metrics import accuracy_score
from rouge_score import rouge_scorer

acc_task_list = ['task363','task875','task1687']

def compute_rouge_l_multiple(references, predictions):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(references, predictions)]
    return sum(scores) / len(scores) if scores else 0.0

def cal_rouge_score(ground_truth_list, predictions_list):
    assert len(ground_truth_list) == len(predictions_list)
    rouge_l_score = compute_rouge_l_multiple(ground_truth_list, predictions_list)
    #print()
    return rouge_l_score


def cal_accuracy(ground_truth_list, predictions_list):
    assert len(ground_truth_list) == len(predictions_list)
    accuracy = accuracy_score(ground_truth_list, predictions_list)
    return accuracy
    

def main(args):
    dataset_order = get_dataset_order(args.dataset_id)
    # 遍历15个service，计算当前dataset id下的average JGA值

    output_dir = os.path.join("./output", args.test_data_name)
    if not os.path.exists(output_dir):
        print(f"results dir {output_dir} not find!")
        sys.exit(1)
    
    avgPerf_list = []
    service_name_list = []
    #print(f"predict result dir: {output_dir}")
    #print("Calculating JGA score for each service.....")
    acc_task_num = 0
    for service_id in range(0, len(dataset_order)):
        task_id = dataset_order[service_id]
        print(f"task name: {dataset_order[service_id]}")
        result_file = os.path.join(output_dir, str(service_id)+"-"+dataset_order[service_id] +"_result.txt")        
        model_results = open(result_file, "r").readlines()
        
        testfile_name = "./data_superni/test/" + dataset_order[service_id] + "_T5.json"
        test_lines = json.load(open(testfile_name))
        
        ground_truth_list = []
        predictions_list = []
        for idx_ in range(0, len(test_lines)):
            ground_truth = test_lines[idx_]['output']
            result_line = model_results[idx_].strip().lower()
            prediction = result_line.split("|||")[-1]
            print(idx_)
            prediction = eval(prediction)[0]
            sample_id = result_line.split("|||")[0]
            if "</s>" in prediction:
                prediction = prediction.replace("</s>","")
            
            if task_id == "task363":
                if prediction == "pos":
                    prediction = "p"
                    
                elif prediction == "neg":
                    prediction = "n"

            
            if test_lines[idx_]['id'].lower() != sample_id:
                print("行没对齐！")
                print(sample_id,test_lines[idx_]['id'] )
                sys.exit(1)
            ground_truth_list.append(ground_truth.lower())
            predictions_list.append(prediction.lower())
            # print(ground_truth)
            # print(prediction)
            # sys.exit(1)
        

        if task_id in acc_task_list:
            acc_task_num += 1
            joint_accuracy = cal_accuracy(ground_truth_list, predictions_list)
            print(f"分类问题的结果是：{joint_accuracy}")
        else:
            joint_accuracy = cal_rouge_score(ground_truth_list, predictions_list)

        #print(ground_truth_list[0], predictions_list[0])
        #print(f"分类问题的结果是：{joint_accuracy}")


        avgPerf_list.append(round(joint_accuracy,4))
        #break    
    print(f"average Performance is {sum(avgPerf_list) / len(avgPerf_list)}")
    print()
    print(f"acc task number is {acc_task_num}, it should be 3.")
    average_JGA = sum(avgPerf_list) / len(avgPerf_list)
    print(avgPerf_list)
    # avgPerf_list.append(average_JGA)
    # service_name_list = dataset_order
    # service_name_list.append("Average")
    # dataframe = pd.DataFrame({'service_name':service_name_list,'avgPerf_list score':avgPerf_list})
    # dataframe.to_csv("./csv_files/avgJGA_dataset_id_"+str(args.dataset_id)+".csv",index=True)
    return average_JGA
            
if __name__=='__main__':
    mean_list = []
    for data_id in range(1):
        
        parser = argparse.ArgumentParser()
        # 所有可能用到的参数
        parser.add_argument("--dataset_id", default=data_id+7, type=int)

 
        parser.add_argument("--test_data_name", type=str, default="t5largelora_SuperNI_Ours_6_inner_step_8_outer_step_8_coldstartstep_3_dataset_id_7_avgPerf", help = "-averaging")
        #parser.add_argument("--test_data_name", type=str, default="t5largelora_Vanilla_inner_step_50000_outer_step_50000_coldstartstep_3_dataset_id_7", help = "-averaging")
        #parser.add_argument("--test_data_name", type=str, default="t5largelora_SuperNI_multitask_dataset_id_7_avgPerf", help = "-averaging")
        

        args = parser.parse_args()
        average_JGA = main(args)
        mean_list.append(average_JGA)
    import numpy as np
    print(np.mean(mean_list))