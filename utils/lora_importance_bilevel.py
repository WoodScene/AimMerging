import math
import torch
import sys

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
        quantile:float,  # 01mask转换
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
        self.quantile = quantile
        print(f"self.taylor is: {self.taylor}")
        print(f"self.quantile is: {self.quantile}")

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
                    #sys.exit(1) 
                    break
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
                
    # 这个应该是最后训练完返回结果
    # 在局部的lora adapter中设置阈值
    def calculate_score_inner_local(self, p=None, metric="ipt"):
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
        

        inner_mask = {}
        for n, score in ipt_score_dic_inner.items():
            #print(n, score)
            # 根据分位数计算 01 mask，将分位数大于 0.5 的元素设为 1，其余设为 0
            threshold = torch.quantile(score, self.quantile)
            inner_mask[n] = (score > threshold).float()
            #print("after 01mask")
            #print(n, score)
        return inner_mask

    # 全局的设置阈值，即遍历所有参数的重要性之后再进行划分
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
        
        # Step 1: 将所有参数的重要性合并成一个大的张量
        all_scores = torch.cat([score.flatten() for score in ipt_score_dic_inner.values()])

        # Step 2: 计算全局的分位数
        threshold = torch.quantile(all_scores, self.quantile)

        inner_mask = {}
        for n, score in ipt_score_dic_inner.items():
            
            inner_mask[n] = (score > threshold).float()
            #print("after 01mask")
            #print(n, score)
        return inner_mask

    def calculate_score_outer_local(self, p=None, metric="ipt"):
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
        # print(f"ipt is {self.ipt}")
        # print(f"exp_avg_ipt is {self.exp_avg_ipt[n]}")
        # print(f"exp_avg_unc is {self.exp_avg_unc[n]}")
        

        outer_mask = {}
        for n, score in ipt_score_dic_outer.items():
            #print(n, score)
            # 根据分位数计算 01 mask，将分位数大于 0.5 的元素设为 1，其余设为 0
            threshold = torch.quantile(score, self.quantile)
            outer_mask[n] = (score > threshold).float()
            #print("after 01mask")
            #print(n, score)
        return outer_mask
    
    def calculate_score_outer(self, p=None, metric="ipt"):
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
        # print(f"ipt is {self.ipt}")
        # print(f"exp_avg_ipt is {self.exp_avg_ipt[n]}")
        # print(f"exp_avg_unc is {self.exp_avg_unc[n]}")
        
        # Step 1: 将所有参数的重要性合并成一个大的张量
        all_scores = torch.cat([score.flatten() for score in ipt_score_dic_outer.values()])

        # Step 2: 计算全局的分位数
        threshold = torch.quantile(all_scores, self.quantile)

        outer_mask = {}
        for n, score in ipt_score_dic_outer.items():

            outer_mask[n] = (score > threshold).float()
            #print("after 01mask")
            #print(n, score)
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
