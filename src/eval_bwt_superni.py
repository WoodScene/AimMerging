
import os
import sys
import json
from glob import glob
import argparse
from utils.dataset_order import get_dataset_order
from sklearn.metrics import accuracy_score
from rouge_score import rouge_scorer

acc_task_list = ['task363','task875','task195']
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

def get_jga_scores(output_dir, dataset_order):
    JGA_list = []
    #print("Calculating JGA score for each service.....")
    acc_task_num = 0
    for service_id in range(0, len(dataset_order)-1):
        #print(service_id)
        result_file = os.path.join(output_dir, str(service_id)+"-"+dataset_order[service_id] +"_result.txt")        
        if not os.path.exists(output_dir):
            print(f"result_file {result_file} not find!")
            sys.exit(1)
        model_results = open(result_file, "r").readlines()
        
        testfile_name = "./data_superni/test/" + dataset_order[service_id] + "_T5.json"
        test_lines = json.load(open(testfile_name))
        
        print(result_file)
        ground_truth_list = []
        predictions_list = []
        for idx_ in range(0, len(model_results)):
            ground_truth = test_lines[idx_]['output']
            result_line = model_results[idx_].strip().lower()
            prediction = result_line.split("|||")[-1]
            prediction = eval(prediction)[0]
            sample_id = result_line.split("|||")[0]
            if "</s>" in prediction:
                prediction = prediction.replace("</s>","")
                
            if test_lines[idx_]['id'].lower() != sample_id:
                print("行没对齐！")
                print(sample_id,test_lines[idx_]['id'] )
                sys.exit(1)
            ground_truth_list.append(ground_truth.lower())
            predictions_list.append(prediction.lower())   
        
        task_id = dataset_order[service_id]

        if task_id in acc_task_list:
            acc_task_num += 1
            joint_accuracy = cal_accuracy(ground_truth_list, predictions_list)
            print(f"分类问题的结果是：{joint_accuracy}")
        else:
            joint_accuracy = cal_rouge_score(ground_truth_list, predictions_list)

        JGA_list.append(joint_accuracy)
    print(f"acc task number is {acc_task_num}, it should be 3.")
    return JGA_list

def main(args):
    dataset_order = get_dataset_order(args.dataset_id)
    # 遍历15-1个service，计算当前dataset id下的average JGA值
    print(f"dataset_order: {dataset_order}")

    output_dir = os.path.join("./output", args.test_data_name,)
    if not os.path.exists(output_dir):
        print(f"results dir {output_dir} not find!")
        sys.exit(1)
    
    # 获取公式中的第一部分结果      
  
    output_dir2 = os.path.join("./output", args.test_data_name2 ,)
    if not os.path.exists(output_dir2):
        print(f"results dir2 {output_dir2} not find!")
        sys.exit(1)
    
    
    avgPerf_list1 = get_jga_scores(output_dir, dataset_order) # Houyibufen
    print(f"avgPerf_list1: {avgPerf_list1}")
    avgPerf_list2 = get_jga_scores(output_dir2, dataset_order)
    print(f"avgPerf_list2: {avgPerf_list2}")
    avgPerf_list = [avgPerf_list2[i]-avgPerf_list1[i] for i in range(len(avgPerf_list1))]
    print(avgPerf_list)
    print(f"BWT is {sum(avgPerf_list) / len(avgPerf_list)}")
    print()
    average_JGA = sum(avgPerf_list) / len(avgPerf_list)
    
    avgPerf_list.append(average_JGA)
    dataset_order.pop()
    dataset_order.append("Average")
    import pandas as pd
    #dataframe = pd.DataFrame({'service_name':dataset_order,'JGA score':JGA_list})
    
    #dataframe.to_csv("./csv_files/bwt_dataset_id_"+str(args.dataset_id)+".csv",index=True)
    return average_JGA

            
if __name__=='__main__':


    dataset_id = 7
    model_name = "t5largelora"


    inner_step = 8
    outer_step = 8
    method_name = f"SuperNI_Ours_6_inner_step_{inner_step}_outer_step_{outer_step}_coldstartstep_3"

    print(method_name)
    parser = argparse.ArgumentParser()
    # 所有可能用到的参数
    parser.add_argument("--dataset_id", default=dataset_id, type=int)

    parser.add_argument("--test_data_name", type=str, default=model_name+"_"+ method_name +"_dataset_id_"+str(dataset_id)+"_bwt", help = "-averaging")
    parser.add_argument("--test_data_name2", type=str, default=model_name+"_"+ method_name +"_dataset_id_"+str(dataset_id)+"_avgPerf", help = "-averaging")
    

    args = parser.parse_args()
    average_JGA = main(args)
 
