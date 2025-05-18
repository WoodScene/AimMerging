import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.dataset_order import get_dataset_order

from scipy import stats

from sklearn.preprocessing import MinMaxScaler

# 把learning signal的变化 （parameters change） 和 forgetting signal的变化 （loss） 放到一个图里


model_name = "t5largelora"
# method_name_ours = "Ours_6_inner_step_8_outer_step_8_coldstartstep_3"
method_name_ours = "SuperNI_Ours_6_inner_step_8_outer_step_8_coldstartstep_3"

dataset_id = 7
for task_id in range(1,15):
    # task_id = 2

    dataset_order = get_dataset_order(dataset_id)


    csv_dir_ours = os.path.join("./csv_file", model_name + "_"+ method_name_ours +"_dataset_id_"+str(dataset_id))        
    csv_file_path_ours = os.path.join(csv_dir_ours, str(task_id) + "-" + dataset_order[task_id] + "_loss.csv")
    df_ours = pd.read_csv(csv_file_path_ours)
    # 打印数据的前几行来检查是否读取成功
    print(df_ours.head())
    history_loss_list_ours = df_ours['History Loss'].tolist()
    history_loss_step_list_ours = df_ours['Step'].tolist()



    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'


    ax1.scatter(history_loss_step_list_ours, history_loss_list_ours, marker='o', alpha=0.5, s=20, color='#f16c23', label='Forgetting Signal')
    

    # # 拟合直线
    # coeffs = np.polyfit(history_loss_step_list_ours, history_loss_list_ours, 3)  # 1 表示拟合直线
    # polynomial = np.poly1d(coeffs)
    # x_fit = np.linspace(min(history_loss_step_list_ours), max(history_loss_step_list_ours), 100)
    # y_fit = polynomial(x_fit)
    # # 绘制拟合直线
    # ax1.plot(x_fit, y_fit, linestyle='-', color='#f16c23', linewidth=3, label='Fitted Line')

    # 设置左侧Y轴标签
    ax1.set_xlabel('Training Step', fontdict={'family' : 'Times New Roman','size': 24})
    ax1.set_ylabel('Forgetting Signal', color='#e87a25', fontdict={'family' : 'Times New Roman','size': 24})
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelcolor='#e87a25', labelsize=14)

    # plt.figure(figsize=(10, 6))
    # plt.scatter(history_loss_step_list_ours, history_loss_list_ours, marker='o', alpha=0.4, s=20, color='#f16c23', label='Forgetting Signal')
    # # 拟合直线
    # coeffs = np.polyfit(history_loss_step_list_ours, history_loss_list_ours, 1)  # 1 表示拟合直线
    # polynomial = np.poly1d(coeffs)
    # # 生成拟合直线的 x 值
    # x_fit = np.linspace(min(history_loss_step_list_ours), max(history_loss_step_list_ours), 100)
    # y_fit = polynomial(x_fit)
    # # 绘制拟合直线
    # plt.plot(x_fit, y_fit, linestyle='-', color='#f16c23', linewidth=3, label='Fitted Line (Ours)')

    # 再画learning signal
    ax2 = ax1.twinx()
    csv_file_path_ours = os.path.join(csv_dir_ours, str(task_id) + "-" + dataset_order[task_id] + ".csv")
    df_ours = pd.read_csv(csv_file_path_ours)
    # 打印数据的前几行来检查是否读取成功
    print(df_ours.head())
    learning_list_ours = df_ours['Delta Change'].tolist()
    step_list_ours = df_ours['Step'].tolist()
    inner_step_list = df_ours['Inner Step'].tolist()

#    ax2.plot(step_list_ours, learning_list_ours, marker='o', linestyle='-', color='b', label='Learning Signal')
    ax2.plot(step_list_ours, learning_list_ours, marker='o', linestyle='--', color='#1883c9', markersize=8, linewidth=2, label='Learning Signal')

    # 设置右侧Y轴标签
    ax2.set_ylabel('Learning Signal', color='#1883c9', fontdict={'family' : 'Times New Roman','size': 24})
    ax2.tick_params(axis='y', labelcolor='#1883c9', labelsize=14)

    # 添加图例
    # fig.tight_layout()  # 自动调整子图参数
    font1 = {'size': 19}
    ax1.legend(loc='upper right', frameon=True, prop=font1, bbox_to_anchor=(1.0, 1))
    ax2.legend(loc='upper right', frameon=True, prop=font1, bbox_to_anchor=(1.0, 0.9))

    # 设置标题和标签
    # plt.title('Changes in Learning and Forgetting Signals during the Training Process', fontname='Times New Roman', fontsize=20)
        
    title_name = f"Changes in Learning and Forgetting Signals (Task id: {task_id + 1}, Task Name: {dataset_order[task_id]})"
    
    plt.title(title_name, fontname='Times New Roman', fontsize=20)

    plt.grid(True)
    # plt.legend()

    # 保存图像
    fig_save_name = str(task_id) + "-" + dataset_order[task_id] + "_signal.png"

    output_dir = os.path.join("./Fig_paper", model_name + "_"+ method_name_ours +"_dataset_id_"+str(dataset_id))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, fig_save_name), dpi=600)
    plt.close()

    # 再画我们的调整策略
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'


#    ax1.plot(step_list_ours, inner_step_list, marker='o', linestyle='-', color='b', label='Merge Contorller')
    ax1.plot(step_list_ours, inner_step_list, marker='^', linestyle='-', color='black', markersize=10, linewidth=2, label='Merge Controller')
    
    title_name = f"Merge Interval Set by the Merge Controller (Task id: {task_id + 1}, Task Name: {dataset_order[task_id]})"
    
    # plt.title('Real-time Fusion Step Size of the Merge Controller', fontname='Times New Roman', fontsize=28)
    plt.title(title_name, fontname='Times New Roman', fontsize=20)
    plt.grid(True)
    ax1.set_xlabel('Training Step', fontdict={'family' : 'Times New Roman','size': 24})
    ax1.set_ylabel('Merge Interval', fontdict={'family' : 'Times New Roman','size': 24})
    ax1.tick_params(axis='y', labelsize=14)
    ax1.tick_params(axis='x',  labelsize=14)
    
    # fig.tight_layout()  # 自动调整子图参数
    fig_save_name = str(task_id) + "-" + dataset_order[task_id] + "_signal_contorller.png"

    plt.savefig(os.path.join(output_dir, fig_save_name), dpi=600)
    plt.close()
