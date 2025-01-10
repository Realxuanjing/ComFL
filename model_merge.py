'''
这是有关模型聚合的代码，包括两个函数：
1. update_gNB_model(gNB_model_state_dict, sum_module_state_dict)
2. FedAvg(w),其中w是一个列表,包含了多个模型的参数字典。

'''
import copy
import torch
import torch.nn as nn
def update_gNB_model(gNB_model_state_dict, sum_module_state_dict, channel_pruning=False):
    """
    将sum_module_state_dict中的参数与gNB_model_state_dict进行更新。
    :param gNB_model_state_dict: gNB模型的状态字典
    :param sum_module_state_dict: 聚合后的模块状态字典
    :return: 更新后的gNB_model_state_dict
    """
    if not channel_pruning:
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
    else:
        updated_gNB_model_state_dict = gNB_model_state_dict.copy()  

        for name, param_sum in sum_module_state_dict.items():
            if name in updated_gNB_model_state_dict:
                # print('name', name, 'param_sum', param_sum.shape, 'param_gNB', updated_gNB_model_state_dict[name].shape)
                param_gNB = updated_gNB_model_state_dict[name]
                
                # 判断 sum_module_state_dict 的参数维度是否比原始模型小
                if param_sum.shape == param_gNB.shape:
                    # 如果维度匹配，则直接替换
                    updated_gNB_model_state_dict[name] = param_sum
                else:
                # else param_sum.shape[0] < param_gNB.shape[0]:
                    # 如果 sum_module_state_dict 的维度比原始模型少，进行替换
                    # 只替换 sum_module_state_dict 中有的部分
                    # updated_gNB_model_state_dict[name][:param_sum.shape[0]] = param_sum
                    new_param = param_gNB.clone()  # 保持原始张量，防止直接修改
                    slices = tuple(slice(0, min(s, t)) for s, t in zip(param_sum.shape, param_gNB.shape))
                    # print('slices', slices)
                    # 使用切片替换对应部分
                    new_param[slices] = param_sum
                    updated_gNB_model_state_dict[name] = new_param
                # else:
                #     print(f"Warning: Unexpected size for {name}, not updating.")
            else:
                # 对于没有在 sum_module_state_dict 中的参数，保持不变
                print(f"Warning: Layer {name} is not in gNB_model_state_dict, keeping sum_module's parameter.")
                pass
        
        return updated_gNB_model_state_dict


# FedAvg函数存在一些问题，最终的参数应该和数据集大小有关
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg



def FedPer_merge_models(gnb_model_dict, local_model_dict):
    gnb_state_dict = copy.deepcopy(gnb_model_dict)
    local_state_dict = copy.deepcopy(local_model_dict)
    result = copy.deepcopy(gnb_state_dict)
    for param_name, param_value in gnb_state_dict.items():
        if "classifier" in param_name:  
            result[param_name] = local_state_dict[param_name] 
        else:
            result[param_name] = param_value  
    
    return result