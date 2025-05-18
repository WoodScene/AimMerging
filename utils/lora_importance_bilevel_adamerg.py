import math
import torch
import sys
import numpy as np
from typing import Optional, List 

# 动态的更新内层和外层的参数重要性分布，并且转换成01mask传回去
class RankAllocator(object):
    """
    The RankAllocator for AdaLoRA Model that will be called every training step. 
    Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        model: the model that we apply AdaLoRA to.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter. 
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter. 
    """
    def __init__(
        self, model, 
        init_warmup:int, 
        beta1:float, 
        beta2:float, 
        rank:int,
        tau:float,  # 01mask转换
        total_step:Optional[int]=None, 
        taylor = None, # 表示用几阶梯度来做为重要性指标 param_second, param_first, param_mix
        
    ):

        self.initial_warmup = init_warmup
        self.beta1 = beta1
        self.beta2 = beta2
        self.rank = rank
        self.total_step = total_step

        self.model = model
        self.ipt_outer = {} 
        self.exp_avg_ipt_outer = {}
        self.exp_avg_unc_outer = {}

        self.ipt_inner = {} 
        self.exp_avg_ipt_inner = {}
        self.exp_avg_unc_inner = {}


        self.taylor = taylor
        self.tau = tau
        print(f"self.taylor is: {self.taylor}")
        print(f"self.tau is: {self.tau}")

        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)

    def set_total_step(self, total_step:int): 
        # Set total step number 
        self.total_step = total_step
        # self.initial_warmup = int(self.total_step / 10) + 1
        self.initial_warmup = 0
        print(f"total_step is {self.total_step}, initial_warmup is {self.initial_warmup}")
        assert self.total_step>self.initial_warmup


    def update_ipt_outer(self, model, global_step): 
        for n,p in model.named_parameters():

            if "lora_" in n:
                if torch.isnan(p.grad).any():
                    print(f"{n},外层循环梯度中存在 NaN 值")
                    #print(p.grad)
                    print(f"step is {global_step}")
                    sys.exit(1) 
                if n not in self.ipt_outer:
                    self.ipt_outer[n] = torch.zeros_like(p)
                    self.exp_avg_ipt_outer[n] = torch.zeros_like(p) 
                    self.exp_avg_unc_outer[n] = torch.zeros_like(p) 
                    #print(f"name n is: {n}, dimension is {p.shape}")
                with torch.no_grad():
                    # Calculate sensitivity 
                    self.ipt_outer[n] = (p * p.grad).abs().detach()
                    if self.taylor in ['param_second']:
                        self.ipt_outer[n] = (p * p.grad * p * p.grad).abs().detach()
                    elif self.taylor in ['param_mix']:
                        self.ipt_outer[n] = (p * p.grad - 0.5 * p * p.grad * p * p.grad).abs().detach()

                    # Update sensitivity 
                    self.exp_avg_ipt_outer[n] = self.beta1 * self.exp_avg_ipt_outer[n] + \
                                        (1-self.beta1)*self.ipt_outer[n]
                    # Update uncertainty 
                    self.exp_avg_unc_outer[n] = self.beta2 * self.exp_avg_unc_outer[n] + \
                                        (1-self.beta2)*(self.ipt_outer[n]-self.exp_avg_ipt_outer[n]).abs()


    def update_ipt_inner(self, model, global_step): 
        for n,p in model.named_parameters():

            if "lora_" in n:
                if torch.isnan(p.grad).any():
                    print(f"{n},梯度中存在 NaN 值")
                    #print(p.grad)
                    print(f"step is {global_step}")
                    break 
                if n not in self.ipt_inner:
                    self.ipt_inner[n] = torch.zeros_like(p)
                    self.exp_avg_ipt_inner[n] = torch.zeros_like(p) 
                    self.exp_avg_unc_inner[n] = torch.zeros_like(p) 
                    #print(f"name n is: {n}, dimension is {p.shape}")
                with torch.no_grad():
                    # Calculate sensitivity 
                    self.ipt_inner[n] = (p * p.grad).abs().detach()
                    if self.taylor in ['param_second']:
                        self.ipt_inner[n] = (p * p.grad * p * p.grad).abs().detach()
                    elif self.taylor in ['param_mix']:
                        self.ipt_inner[n] = (p * p.grad - 0.5 * p * p.grad * p * p.grad).abs().detach()

                    # Update sensitivity 
                    self.exp_avg_ipt_inner[n] = self.beta1 * self.exp_avg_ipt_inner[n] + \
                                        (1-self.beta1)*self.ipt_inner[n]
                    # Update uncertainty 
                    self.exp_avg_unc_inner[n] = self.beta2 * self.exp_avg_unc_inner[n] + \
                                        (1-self.beta2)*(self.ipt_inner[n]-self.exp_avg_ipt_inner[n]).abs()
                # print(f"step is {global_step}")
                # print(f"name is {n}")
                # print(f"ipt is {self.ipt[n]}")
                # print()

    def normalize_importance_scores(self, ipt_score_dic):
        """
        对一个重要性字典中的所有重要性值进行全局归一化处理，使得它们的值在 0 到 1 之间。

        Args:
            ipt_score_dic (dict): 重要性字典，其中每个 key 对应一个参数名称，value 是该参数的重要性。

        Returns:
            dict: 归一化后的参数重要性字典。
        """
        # 提取所有重要性值到一个列表中
        all_scores_tensor = torch.cat([score.flatten() for score in ipt_score_dic.values()])


        # 计算最小值和最大值
        min_score = torch.min(all_scores_tensor)
        max_score = torch.max(all_scores_tensor)
        #print(f"min_score: {min_score}")
        #print(f"max_score: {max_score}")
        # 对所有值进行归一化
        normalized_dic = {}
        for n, score in ipt_score_dic.items():
            normalized_dic[n] = (score - min_score) / (max_score - min_score)


        all_scores_tensor = torch.cat([score.flatten() for score in normalized_dic.values()])
        min_score = torch.min(all_scores_tensor)
        max_score = torch.max(all_scores_tensor)
        #print(f"归一化后min_score: {min_score}")
        #print(f"归一化后max_score: {max_score}")
        #sys.exit(1)
        return normalized_dic


    # 这个应该是最后训练完返回结果
    def calculate_score_inner(self, p=None, metric="ipt"):
        assert len(self.exp_avg_ipt_inner) == len(self.exp_avg_unc_inner)
        


        ipt_score_dic_inner = {}
        for n in self.exp_avg_ipt_inner:
            #print(f"name is {n}")
            #ipt_name_list.append(n)
            if metric == "ipt":
                # Combine the senstivity and uncertainty 
                ipt_score = self.exp_avg_ipt_inner[n] * self.exp_avg_unc_inner[n]
            elif metric == "mag":
                ipt_score = p.abs().detach().clone() 
            else:
                raise ValueError("Unexcptected Metric: %s"%metric)
            
            # print(f"score is {ipt_score}")
            #ipt_score_mean = torch.mean(ipt_score).item()
            #print(f"mean score is {ipt_score_mean}")
            
            ipt_score_dic_inner[n] = ipt_score
        # print(f"ipt is {self.ipt}")
        # print(f"exp_avg_ipt is {self.exp_avg_ipt[n]}")
        # print(f"exp_avg_unc is {self.exp_avg_unc[n]}")
        assert len(self.exp_avg_ipt_outer) == len(self.exp_avg_unc_outer)
        
        
        ipt_score_dic_outer = {}
        for n in self.exp_avg_ipt_outer:
            #print(f"name is {n}")
            #ipt_name_list.append(n)
            if metric == "ipt":
                # Combine the senstivity and uncertainty 
                ipt_score = self.exp_avg_ipt_outer[n] * self.exp_avg_unc_outer[n]
            elif metric == "mag":
                ipt_score = p.abs().detach().clone() 
            else:
                raise ValueError("Unexcptected Metric: %s"%metric)
            
            # print(f"score is {ipt_score}")
            #ipt_score_mean = torch.mean(ipt_score).item()
            #print(f"mean score is {ipt_score_mean}")
            
            ipt_score_dic_outer[n] = ipt_score

        # 补充：需要做一个归一化操作，否则会导致权重都是0.5
        # 原因是重要性的值都非常小，可能有e-15次方左右，做完指数操作后，结果都是1
        
        # 新增全局01归一化操作
        ipt_score_dic_inner_norm = self.normalize_importance_scores(ipt_score_dic_inner)
        ipt_score_dic_outer_norm = self.normalize_importance_scores(ipt_score_dic_outer)

        inner_mask = {}

        for key in ipt_score_dic_inner_norm:
            assert key in ipt_score_dic_outer_norm
            ipt_score_inner = ipt_score_dic_inner_norm[key].cpu().numpy()
            ipt_score_outer = ipt_score_dic_outer_norm[key].cpu().numpy()

            exp_term_inner = np.exp(ipt_score_inner / self.tau)
            exp_term_outer = np.exp(ipt_score_outer / self.tau)
            denominator = exp_term_inner + exp_term_outer
            
            coefficient_inner = exp_term_inner / denominator

            inner_mask[key] = coefficient_inner
            inner_mask[key] = torch.tensor(coefficient_inner, device=ipt_score_dic_inner[key].device)
 
 
        return inner_mask

    def calculate_score_outer(self, p=None, metric="ipt"):
        assert len(self.exp_avg_ipt_inner) == len(self.exp_avg_unc_inner)

        ipt_score_dic_inner = {}
        for n in self.exp_avg_ipt_inner:
            #print(f"name is {n}")
            #ipt_name_list.append(n)
            if metric == "ipt":
                # Combine the senstivity and uncertainty 
                ipt_score = self.exp_avg_ipt_inner[n] * self.exp_avg_unc_inner[n]
            elif metric == "mag":
                ipt_score = p.abs().detach().clone() 
            else:
                raise ValueError("Unexcptected Metric: %s"%metric)
            
            # print(f"score is {ipt_score}")
            #ipt_score_mean = torch.mean(ipt_score).item()
            #print(f"mean score is {ipt_score_mean}")
            
            ipt_score_dic_inner[n] = ipt_score
        # print(f"ipt is {self.ipt}")
        # print(f"exp_avg_ipt is {self.exp_avg_ipt[n]}")
        # print(f"exp_avg_unc is {self.exp_avg_unc[n]}")
        assert len(self.exp_avg_ipt_outer) == len(self.exp_avg_unc_outer)
        
        
        ipt_score_dic_outer = {}
        for n in self.exp_avg_ipt_outer:
            #print(f"name is {n}")
            #ipt_name_list.append(n)
            if metric == "ipt":
                # Combine the senstivity and uncertainty 
                ipt_score = self.exp_avg_ipt_outer[n] * self.exp_avg_unc_outer[n]
            elif metric == "mag":
                ipt_score = p.abs().detach().clone() 
            else:
                raise ValueError("Unexcptected Metric: %s"%metric)
            
            # print(f"score is {ipt_score}")
            #ipt_score_mean = torch.mean(ipt_score).item()
            #print(f"mean score is {ipt_score_mean}")
            
            ipt_score_dic_outer[n] = ipt_score

        #print(f"ipt_score_dic_inner: {ipt_score_dic_inner}")
        #print(f"ipt_score_dic_outer: {ipt_score_dic_outer}")
        #sys.exit(1)

        # 补充：需要做一个归一化操作，否则会导致权重都是0.5
        # 原因是重要性的值都非常小，可能有e-15次方左右，做完指数操作后，结果都是1
        
        # 新增全局01归一化操作
        ipt_score_dic_inner_norm = self.normalize_importance_scores(ipt_score_dic_inner)
        ipt_score_dic_outer_norm = self.normalize_importance_scores(ipt_score_dic_outer)

        outer_mask = {}

        for key in ipt_score_dic_inner_norm:
            assert key in ipt_score_dic_outer_norm
            ipt_score_inner = ipt_score_dic_inner_norm[key].cpu().numpy()
            ipt_score_outer = ipt_score_dic_outer_norm[key].cpu().numpy()
            #print(f"ipt_score_inner: {ipt_score_inner}")
            #print(f"ipt_score_outer: {ipt_score_outer}")
            exp_term_inner = np.exp(ipt_score_inner / self.tau)
            exp_term_outer = np.exp(ipt_score_outer / self.tau)
            #print(f"exp_term_inner: {exp_term_inner}")
            #print(f"exp_term_outer: {exp_term_outer}")
            #sys.exit(1)
            denominator = exp_term_inner + exp_term_outer
            
            coefficient_outer = exp_term_outer / denominator

            outer_mask[key] = torch.tensor(coefficient_outer, device=ipt_score_dic_outer[key].device)
 

        return outer_mask
    
    def update_inner_score(self, model, global_step):
        # if global_step < self.total_step and global_step > self.initial_warmup:
        #     # Update importance scores element-wise 
        #     self.update_ipt(model, global_step)

        self.update_ipt_inner(model, global_step)
    
    def update_outer_score(self, model, global_step):
        # if global_step < self.total_step and global_step > self.initial_warmup:
        #     # Update importance scores element-wise 
        #     self.update_ipt(model, global_step)

        self.update_ipt_outer(model, global_step)


    def empty_inner_score(self):
        self.ipt_inner = {} 
        self.exp_avg_ipt_inner = {}
        self.exp_avg_unc_inner = {}

    def empty_outer_score(self):
        self.ipt_outer = {} 
        self.exp_avg_ipt_outer = {}
        self.exp_avg_unc_outer = {}
