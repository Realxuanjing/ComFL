'''
这是有关模型聚合的代码，包括两个函数：
1. update_gNB_model(gNB_model_state_dict, sum_module_state_dict)
2. FedAvg(w),其中w是一个列表,包含了多个模型的参数字典。

'''
import copy
import torch
import torch.nn as nn
def update_gNB_model(gNB_model_state_dict, sum_module_state_dict):
    """
    将sum_module_state_dict中的参数与gNB_model_state_dict进行更新。
    :param gNB_model_state_dict: gNB模型的状态字典
    :param sum_module_state_dict: 聚合后的模块状态字典
    :return: 更新后的gNB_model_state_dict
    """
    updated_gNB_model_state_dict = gNB_model_state_dict.copy()  

    for name, param_gNB in gNB_model_state_dict.items():
        if name in sum_module_state_dict:
            param_sum = sum_module_state_dict[name]
            
            # 检查两者的形状是否一致，确保可以进行平均
            if param_gNB.shape == param_sum.shape:
                # print(f"Updating layer {name}: average of {param_gNB.shape}")
                updated_gNB_model_state_dict[name] =  param_sum
            else:
                pass
                # print(f"Layer {name} has mismatched shapes, keeping gNB_model's parameter.")
        else:
            pass
            # print(f"Layer {name} is not in sum_module, keeping gNB_model's parameter.")
    
    return updated_gNB_model_state_dict


# FedAvg函数存在一些问题，最终的参数应该和数据集大小有关
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg