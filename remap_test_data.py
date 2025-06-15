import os
import pandas as pd
import shutil

def remap_test_data():
    # 输入和输出文件路径
    test_file = '/home/zh/cxy/code/IDCE_main/datasets/ml-1m/LightGCN_ml-1m-dataset_test.txt'
    user_map_file = '/home/zh/cxy/code/IDCE_main/pretrained_scores/ml-1m/user_map_LightGCN_ml-1m.txt'
    temp_file = '/home/zh/cxy/code/IDCE_main/datasets/ml-1m/LightGCN_ml-1m-dataset_test_tmp.txt'
    
    # 检查文件是否存在
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    if not os.path.exists(user_map_file):
        print(f"用户映射文件不存在: {user_map_file}")
        return
    
    # 加载用户映射 (行索引 -> 用户ID)
    user_map_df = pd.read_csv(user_map_file, header=None, names=['row_idx', 'user_id'])
    
    # 创建反向映射 (用户ID -> 行索引)
    reverse_map = {user_id: row_idx for row_idx, user_id in zip(user_map_df['row_idx'], user_map_df['user_id'])}
    
    # 读取并处理测试数据
    with open(test_file, 'r') as f:
        test_lines = f.readlines()
    
    # 写入重映射后的数据到临时文件
    with open(temp_file, 'w') as f:
        for line in test_lines:
            parts = line.strip().split(',')
            if len(parts) != 2:
                continue
                
            old_user_id, item_id = int(parts[0]), parts[1]
            
            # 检查用户ID是否在映射中
            if old_user_id in reverse_map:
                new_user_id = reverse_map[old_user_id]
                f.write(f"{new_user_id},{item_id}\n")
            else:
                print(f"警告: 用户ID {old_user_id} 不在映射中，跳过")
    
    # 统计处理结果
    count_original = len(test_lines)
    with open(temp_file, 'r') as f:
        count_remapped = len(f.readlines())
    
    # 备份原文件并替换
    backup_file = f"{test_file}.bak"
    shutil.copy2(test_file, backup_file)
    shutil.move(temp_file, test_file)
    
    print(f"重映射完成! 原始文件已被修改: {test_file}")
    print(f"原始文件已备份到: {backup_file}")
    print(f"原始记录数: {count_original}")
    print(f"重映射记录数: {count_remapped}")
    
    if count_original != count_remapped:
        print(f"警告: 部分记录未被映射 ({count_original - count_remapped}条)")

if __name__ == '__main__':
    remap_test_data() 