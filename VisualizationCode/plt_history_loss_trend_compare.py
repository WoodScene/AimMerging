import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.dataset_order import get_dataset_order

from scipy import stats

import matplotlib.ticker as ticker

from sklearn.preprocessing import MinMaxScaler

# 把正常训练和做了model merge的loss 变化放到一个图里，方便对比；不要做归一化了，这样原始的数值大小也可以对比

model_name = "t5largelora"

# method_name_ours = "Ours_5_2_inner_step_8_outer_step_8_coldstartstep_3"

method_name_ours = "Ours_inner_step_8_outer_step_8_coldstartstep_3"
# method_name_ours = "SuperNI_Ours_6_inner_step_8_outer_step_8_coldstartstep_3"


method_name_vanilla = "Vanilla_inner_step_50000_outer_step_50000_coldstartstep_3"
# method_name_vanilla = "Ours_4_inner_step_320_outer_step_320_coldstartstep_3"



dataset_id = 1
#task_id = 2
for task_id in range(1,15):
        
    dataset_order = get_dataset_order(dataset_id)


    csv_dir_ours = os.path.join("./csv_file", model_name + "_"+ method_name_ours +"_dataset_id_"+str(dataset_id))        
    csv_file_path_ours = os.path.join(csv_dir_ours, str(task_id) + "-" + dataset_order[task_id] + "_loss.csv")
    df_ours = pd.read_csv(csv_file_path_ours)
    # 打印数据的前几行来检查是否读取成功
    print(df_ours.head())
    history_loss_list_ours = df_ours['History Loss'].tolist()
    history_loss_step_list_ours = df_ours['Step'].tolist()

    csv_dir_vanilla = os.path.join("./csv_file", model_name + "_"+ method_name_vanilla +"_dataset_id_"+str(dataset_id))        
    csv_file_path_vanilla = os.path.join(csv_dir_vanilla, str(task_id) + "-" + dataset_order[task_id] + "_loss.csv")
    df_vanilla = pd.read_csv(csv_file_path_vanilla)
    # 打印数据的前几行来检查是否读取成功
    print(df_vanilla.head())
    history_loss_list_vanilla = df_vanilla['History Loss'].tolist()
    history_loss_step_list_vanilla = df_vanilla['Step'].tolist()

    assert history_loss_step_list_vanilla == history_loss_step_list_ours


    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.scatter(history_loss_step_list_vanilla, history_loss_list_vanilla, marker='o', alpha=0.4, s=20, color='#2b6a99', label='History Loss (Vanilla)')
    plt.scatter(history_loss_step_list_ours, history_loss_list_ours, marker='o', alpha=0.4, s=20, color='#f16c23', label='History Loss (Ours)')

    # 拟合直线
    coeffs = np.polyfit(history_loss_step_list_vanilla, history_loss_list_vanilla, 1)  # 1 表示拟合直线
    polynomial = np.poly1d(coeffs)
    # 生成拟合直线的 x 值
    x_fit = np.linspace(min(history_loss_step_list_vanilla), max(history_loss_step_list_vanilla), 100)
    y_fit = polynomial(x_fit)
    # 绘制拟合直线
    plt.plot(x_fit, y_fit, linestyle='-', color='#2b6a99', linewidth=3, label='Fitted Line (Vanilla)')

    # 拟合直线
    coeffs = np.polyfit(history_loss_step_list_ours, history_loss_list_ours, 1)  # 1 表示拟合直线
    polynomial = np.poly1d(coeffs)
    # 生成拟合直线的 x 值
    x_fit = np.linspace(min(history_loss_step_list_ours), max(history_loss_step_list_ours), 100)
    y_fit = polynomial(x_fit)
    # 绘制拟合直线
    plt.plot(x_fit, y_fit, linestyle='-', color='#f16c23', linewidth=3, label='Fitted Line (Ours)')


    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x',  labelsize=14)
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')

    # 设置标题和标签
    title_name = f"Variation of Historical Loss During Training (Task id: {task_id + 1}, Task Name: {dataset_order[task_id]})"
    plt.title(title_name, fontname='Times New Roman', fontsize=20)
    plt.xlabel('Training Step', fontdict={'family' : 'Times New Roman','size': 20})
    plt.ylabel('Hisotry Loss', fontdict={'family' : 'Times New Roman','size': 20})
    plt.grid(True)
    fig.tight_layout()  # 自动调整子图参数
    font1 = {'size': 15}
    plt.legend(loc='upper left',prop=font1)

    # Your plot code here

    # Enable scientific notation on the y-axis
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.gca().yaxis.get_major_formatter().set_scientific(True)

    # 保存图像

    output_dir = os.path.join("./Fig_paper", "Compare_" + model_name + "_"+ method_name_ours +"_dataset_id_"+str(dataset_id))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig_save_name = str(task_id) + "-" + dataset_order[task_id] + "_historyloss_compare.png"


    plt.savefig(os.path.join(output_dir, fig_save_name), dpi=600)
    plt.close()
