import os
import torch
import numpy as np
import pandas as pd
import argparse
from recbole.data import create_samplers, data_preparation
from recbole.utils import get_model, get_trainer, init_seed
from recbole.config import Config
from recbole.data.dataset import Dataset
from recbole.utils.case_study import full_sort_scores

# 导入我们自定义的模型
from xsimgcl import XSimGCL
from supccl import SUPCCL

def generate_scores(model_name='LightGCN', dataset_name='ml-1m'):
    """
    根据指定的模型和数据集生成预测评分
    
    Args:
        model_name: 推荐模型名称 (LightGCN, SimpleX, SGL, NCL, XSimGCL, SUPCCL)
        dataset_name: 数据集名称 (ml-1m, amazon-beauty, amazon-toys-games)
    """
    print(f"使用模型 {model_name} 在数据集 {dataset_name} 上生成预测评分")
    
    # 数据集名称标准化
    if dataset_name.lower() == 'ml-1m' or dataset_name.lower() == 'movielens-1m':
        recbole_dataset_name = 'ml-1m'
        dataset_dir = 'ml-1m'
    elif dataset_name.lower() == 'amazon-beauty':
        recbole_dataset_name = 'amazon-beauty'
        dataset_dir = 'amazon-beauty'
    elif dataset_name.lower() == 'amazon-toys-games':
        recbole_dataset_name = 'amazon-toys-games'
        dataset_dir = 'amazon-toys-games'
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 模型名称标准化
    valid_models = ['LightGCN', 'SimpleX', 'SGL', 'NCL', 'XSimGCL', 'SUPCCL']
    if model_name not in valid_models:
        raise ValueError(f"不支持的模型: {model_name}，支持的模型有: {', '.join(valid_models)}")
    
    # 设置参数 - 使用RecBole推荐的超参数
    parameter_dict = {
        'seed': 2024,
        'gpu_id': 0,
        'use_gpu': True,
        'train_batch_size': 2048,
        'eval_batch_size': 4096,
        'learning_rate': 0.001,  # 学习率
        'epochs': 100,  
        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},  # 明确指定要加载的列
        'field_separator': "\t",  # 指定字段分隔符为制表符
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'TIME_FIELD': 'timestamp',  # 指定时间戳字段
        'USE_FIELD_SEPARATOR': True,  # 使用字段分隔符
        'eval_args': {
            'split': {'RS': [0.7, 0.1, 0.2]},
            'order': 'TO',
            # 'group_by': 'user',
            # 'mode': 'full',
        },
        'save_dataset': True,
        'save_dataloaders': False,
        'metrics': ['Recall', 'NDCG', 'MRR','Precision'],
        'valid_metric': 'NDCG@10',
        'topk': [5, 10, 20, 50],
        'data_path': 'recbole/dataset',  # 指定本地数据集路径
        'log_wandb': False,
        'log_tensorboard': False,
        'stopping_step': 10,
        'state': 'INFO',  # 设置日志级别为INFO，显示完整配置
    }
    
    # 根据不同模型添加特定的超参数
    if model_name == 'LightGCN':
        print("LightGCN")
        parameter_dict.update({
            'n_layers': 1,
            'reg_weight': 1e-02,
            'learning_rate': 5e-4,
        })
    elif model_name == 'SimpleX':
        parameter_dict.update({
            'gamma': 0.7,
            'embedding_size': 64,
            'margin': 0.9,
            'negative_weight': 50,
        })
    elif model_name == 'SGL':
        parameter_dict.update({
            'n_layers': 3,
            'ssl_temp': 0.5,  
            'ssl_reg': 0.05,  
            'dropout': 0.1,
            'type': 'ED',      
        })
    elif model_name == 'NCL':
        parameter_dict.update({
            'n_layers': 3,
            'temp': 0.1,           # 更新ssl_temp
            'reg_weight': 1e-4,    # 更新reg_weight
            'proto_reg': 8e-8,     # 更新proto_reg
            'num_clusters': 1000,  # 更新num_clusters
            'ssl_reg': 1e-7,       # 更新ssl_reg
            'alpha': 1             # 添加alpha参数
        })
    elif model_name == 'XSimGCL':
        parameter_dict.update({
            'n_layers': 2,
            'lambda_coeff': 0.1,
            'eps': 0.2,
            'tau': 0.2,
            'reg_weight': 0.0001,
            'layer_cl': 1,
            'learning_rate': 0.002,
            'embedding_size': 64,
        })
    elif model_name == 'SUPCCL':
        parameter_dict.update({
            'n_layers': 2,
            'embedding_size': 64,
            'learning_rate': 0.001,
            'reg_weight': 0.0001,
            'ssl_temp': 0.1,
            'ssl_reg': 0.3,
            'ssl_ratio': 0.3,
            'ssl_mode': 3,  # merge模式
            'ssl_strategy': 9,
            'aug_type': 1,  # edge dropout
            'positive_cl_type': 1,
            'lightgcn_flag': False,
            'pairwise_loss': False,
            'random_strategy': True,
            'augmentation': True,
            'add_initial_embedding': True,
            'interacted_neighbors': True,
            'similar_user_neighbors': True,
            'similar_item_neighbors': True,
            'different_view': True,
            'different_view_weight': 1.0,
            'interacted_neighbors_weight': 1.0,
            'sample_item_weight': 1.0,
            'sample_user_weight': 1.0,
            'sample_item_weight_flag': False,
            'sample_user_weight_flag': False,
            'supcon_flag': False,
            'prob_sampling': True,
            'sub_graph_pool': 300,
            'k': 5,
            'valid_metric': 'NDCG@20',
            'epochs': 300,
            'train_neg_sample_args': {'by': 1},
            'train_data_step': 4096,
            'val_data_step': 256,
            'test_data_step': 256,
            'neg_sampling': {'uniform': 1},
        })

    # 初始化配置
    if model_name == 'XSimGCL':
        # 使用自定义的XSimGCL模型类
        config = Config(model=XSimGCL, dataset=recbole_dataset_name, config_dict=parameter_dict)
    elif model_name == 'SUPCCL':
        # 使用自定义的SUPCCL模型类
        config = Config(model=SUPCCL, dataset=recbole_dataset_name, config_dict=parameter_dict)
    else:
        # 其他模型使用模型名称字符串
        config = Config(model=model_name, dataset=recbole_dataset_name, config_dict=parameter_dict)
    
    init_seed(config['seed'], config['reproducibility'])

    # 加载数据
    dataset = Dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 导出数据集划分结果到txt文件
    print("正在导出数据集划分结果...")
    
    # 创建输出目录
    output_dir = f'IDCE_main/datasets/{dataset_dir}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取训练集、验证集和测试集的交互数据
    train_inter = train_data.dataset.inter_feat
    valid_inter = valid_data.dataset.inter_feat
    test_inter = test_data.dataset.inter_feat
    
    # 获取用户和物品ID字段名
    uid_field, iid_field = dataset.uid_field, dataset.iid_field
    
    # 导出训练集
    train_path = os.path.join(output_dir, f'{model_name}_{dataset_dir}-dataset_train.txt')
    with open(train_path, 'w') as f:
        for i in range(len(train_inter)):
            user = train_inter[uid_field][i].item()
            item = train_inter[iid_field][i].item()
            f.write(f"{user},{item}\n")
    print(f"训练集已导出到 {train_path}，共 {len(train_inter)} 条记录")
    
    # 导出验证集
    valid_path = os.path.join(output_dir, f'{model_name}_{dataset_dir}-dataset_val.txt')
    with open(valid_path, 'w') as f:
        for i in range(len(valid_inter)):
            user = valid_inter[uid_field][i].item()
            item = valid_inter[iid_field][i].item()
            f.write(f"{user},{item}\n")
    print(f"验证集已导出到 {valid_path}，共 {len(valid_inter)} 条记录")
    
    # 导出测试集
    test_path = os.path.join(output_dir, f'{model_name}_{dataset_dir}-dataset_test.txt')
    with open(test_path, 'w') as f:
        for i in range(len(test_inter)):
            user = test_inter[uid_field][i].item()
            item = test_inter[iid_field][i].item()
            f.write(f"{user},{item}\n")
    print(f"测试集已导出到 {test_path}，共 {len(test_inter)} 条记录")

    # 加载模型
    if model_name == 'XSimGCL':
        # 对于XSimGCL，直接初始化模型类
        model = XSimGCL(config, dataset).to(config['device'])
    elif model_name == 'SUPCCL':
        # 对于SUPCCL，直接初始化模型类
        model = SUPCCL(config, dataset).to(config['device'])
    else:
        # 其他模型使用get_model获取
        model = get_model(config['model'])(config, dataset).to(config['device'])
    
    # 检查是否有对应模型的checkpoint
    checkpoint_file = f'saved/{model_name}-{dataset_dir}.pth'
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        # 检查checkpoint的结构
        print(f"Checkpoint keys: {checkpoint.keys()}")
        
        # 尝试不同的加载方式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model using 'model_state_dict' key")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded model using 'state_dict' key")
        else:
            # 尝试直接加载整个checkpoint
            try:
                model.load_state_dict(checkpoint)
                print("Loaded model directly from checkpoint")
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Training from scratch instead...")
                # 创建训练器
                trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
                
                # 训练模型
                best_valid_score, best_valid_result = trainer.fit(
                    train_data,
                    valid_data,
                    saved=True,
                    show_progress=True,
                    verbose=True
                )
                print(f"Best valid score: {best_valid_score}")
                print(f"Best valid result: {best_valid_result}")
                
                # 在测试集上评估
                test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
                print(f"Test result: {test_result}")
                
                # 保存评估结果
                result_dir = f'IDCE_main/results/{dataset_dir}'
                os.makedirs(result_dir, exist_ok=True)
                
                # 准备保存的结果数据
                save_result = {
                    'model': model_name,
                    'dataset': dataset_name,
                    'best_valid_score': float(best_valid_score),
                    'best_valid_result': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in best_valid_result.items()},
                    'test_result': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in test_result.items()},
                    'parameters': parameter_dict
                }
                
                # 保存为JSON文件
                import json
                result_path = os.path.join(result_dir, f'{model_name}_{dataset_dir}_results.json')
                with open(result_path, 'w') as f:
                    json.dump(save_result, f, indent=4)
                print(f"评估结果已保存至 {result_path}")
    else:
        print("No pretrained model found, training from scratch...")
        # 创建训练器
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        
        # 训练模型
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=True,
            show_progress=True,
            verbose=True
        )
        print(f"Best valid score: {best_valid_score}")
        print(f"Best valid result: {best_valid_result}")
        
        # 在测试集上评估
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
        print(f"Test result: {test_result}")
        
        # 保存评估结果
        result_dir = f'IDCE_main/results/{dataset_dir}'
        os.makedirs(result_dir, exist_ok=True)
        
        # 准备保存的结果数据
        save_result = {
            'model': model_name,
            'dataset': dataset_name,
            'best_valid_score': float(best_valid_score),
            'best_valid_result': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in best_valid_result.items()},
            'test_result': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in test_result.items()},
            'parameters': parameter_dict
        }
        
        # 保存为JSON文件
        import json
        result_path = os.path.join(result_dir, f'{model_name}_{dataset_dir}_results.json')
        with open(result_path, 'w') as f:
            json.dump(save_result, f, indent=4)
        print(f"评估结果已保存至 {result_path}")

    # 设置为评估模式
    model.eval()
    
    # 获取用户和物品的数量
    num_users = dataset.user_num
    num_items = dataset.item_num
    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    
    # 获取测试集中的用户ID
    test_users = torch.unique(test_data.dataset.inter_feat[dataset.uid_field]).tolist()
    num_test_users = len(test_users)
    print(f"Number of test users: {num_test_users}")
    
    # 计算评分矩阵
    print("Calculating scores matrix using full_sort_scores...")
    
    # 分批处理用户，避免显存溢出
    user_ids = torch.tensor(test_users, device='cpu')
    batch_size = 100  # 一次处理100个用户，可以根据显存情况调整
    num_batches = (len(user_ids) + batch_size - 1) // batch_size
    scores_list = []
    
    print(f"将测试用户分为{num_batches}批进行处理，每批{batch_size}个用户")
    
    try:
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(user_ids))
            batch_user_ids = user_ids[batch_start:batch_end]
            
            print(f"处理第{i+1}/{num_batches}批用户，包含{len(batch_user_ids)}个用户")
            
            # 如果显存仍然不足，可以尝试在CPU上计算
            try:
                batch_scores = full_sort_scores(batch_user_ids, model, test_data, device=config['device']).cpu().numpy()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print("GPU内存不足，尝试在CPU上计算...")
                    torch.cuda.empty_cache()  # 清理GPU缓存
                    batch_scores = full_sort_scores(batch_user_ids, model, test_data, device='cpu').cpu().numpy()
                else:
                    raise e
            
            scores_list.append(batch_scores)
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
        # 合并所有批次的结果
        scores_matrix = np.concatenate(scores_list, axis=0)
    except Exception as e:
        print(f"生成评分矩阵时发生错误: {e}")
        print("尝试使用更小的批次大小或在CPU上计算...")
        
        # 如果上面的方法失败，尝试完全在CPU上处理，每次只处理一个用户
        print("使用单用户处理模式...")
        model = model.to('cpu')  # 将模型移动到CPU
        scores_list = []
        
        for i, user_id in enumerate(user_ids):
            if i % 10 == 0:
                print(f"处理用户 {i+1}/{len(user_ids)}")
            user_id_tensor = user_id.unsqueeze(0)
            user_score = full_sort_scores(user_id_tensor, model, test_data, device='cpu').cpu().numpy()
            scores_list.append(user_score)
        
        scores_matrix = np.concatenate(scores_list, axis=0)
        print("已完成所有用户的评分计算")
    
    print(f"Generated scores matrix with shape: {scores_matrix.shape}")
    
    # 处理评分矩阵中的-inf值，替换为-100000000.0
    scores_matrix[scores_matrix == float('-inf')] = -100000000.0
    scores_matrix[np.isneginf(scores_matrix)] = -100000000.0
    
    # 标准化评分到[0,1]范围 - 可选步骤
    if np.min(scores_matrix) < 0 or np.max(scores_matrix) > 1:
        min_score = np.min(scores_matrix[scores_matrix > -1e7]) if np.any(scores_matrix > -1e7) else 0
        max_score = np.max(scores_matrix)
        if max_score > min_score:
            # 只标准化非缺失值
            normal_mask = scores_matrix > -1e7
            scores_matrix[normal_mask] = (scores_matrix[normal_mask] - min_score) / (max_score - min_score)
    
    print(f"处理后的评分矩阵范围: [{np.min(scores_matrix)}, {np.max(scores_matrix)}]")
    print(f"0值比例: {np.mean(scores_matrix == 0):.4f}")
    
    # 保存得分矩阵
    save_dir = f'IDCE_main/pretrained_scores/{dataset_dir}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{model_name}_{dataset_dir}_scores_tensor.txt'
    
    # 使用pandas保存为CSV格式，确保与IDCR.py兼容
    scores_df = pd.DataFrame(scores_matrix)
    scores_df.to_csv(save_path, header=False, index=False)
    print(f"Saved scores matrix to {save_path}")
    print(f"Matrix shape: {scores_matrix.shape}")
    
    # 保存用户ID到矩阵行索引的映射
    user_map_path = f'{save_dir}/user_map_{model_name}_{dataset_dir}.txt'
    with open(user_map_path, 'w') as f:
        for idx, user_id in enumerate(test_users):
            f.write(f"{idx},{user_id}\n")
    print(f"保存了用户ID映射到 {user_map_path}，格式为：行索引,用户ID")
    
    # 验证保存的文件
    try:
        # 尝试使用pandas加载保存的文件
        loaded_df = pd.read_csv(save_path, header=None)
        print(f"Successfully loaded saved matrix with shape: {loaded_df.shape}")
        print(f"Matrix data type: {loaded_df.dtypes[0]}")
    except Exception as e:
        print(f"Error loading saved matrix: {e}")

def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description='生成推荐模型预测评分矩阵')
    
    # 添加模型和数据集参数
    parser.add_argument('--model', type=str, default='LightGCN', 
                        choices=['LightGCN', 'SimpleX', 'SGL', 'NCL', 'XSimGCL', 'SUPCCL'],
                        help='推荐模型名称 (LightGCN, SimpleX, SGL, NCL, XSimGCL, SUPCCL)')
    
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        choices=['ml-1m', 'amazon-beauty', 'amazon-toys-games'],
                        help='数据集名称 (ml-1m, amazon-beauty, amazon-toys-games)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数生成评分
    generate_scores(model_name=args.model, dataset_name=args.dataset)

if __name__ == '__main__':
    main() 