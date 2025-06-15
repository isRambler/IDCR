#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.sparse as sp

# 全局参数
ALPHA = 0.5  # 度中心性权重
BETA = 0.3   # 介数中心性权重
GAMMA = 0.2  # 特征中心性权重
LAMBDA = 0.5 # 影响半径缩放参数
BASE_RADIUS = 0.2 # 基础影响半径
ITEM_LIMIT_NUM = 20 # 每个物品最多选择的用户数
TOPK = 10 # 推荐列表长度
MAX_ITER = 3 # 最大迭代次数

def load_data(data_name='ml-1m', model_name='LightGCN'):
    """
    加载预训练评分矩阵和测试数据
    
    Args:
        data_name: 数据集名称
        model_name: 预训练模型名称
        
    Returns:
        scores_tensor_df: 预训练的评分矩阵
        test_user_list: 测试集，每个用户对应的物品列表
    """
    # 加载预训练评分矩阵
    # scores_tensor_df = pd.read_csv(f'./pretrained_scores/{data_name}/{model_name}_{data_name}_scores_tensor.txt', header=None)
    scores_tensor_df = pd.read_csv('/home/zh/cxy/code/IDCE_main/pretrained_scores/ml-1m/LightGCN_ml-1m_scores_tensor.txt', header=None)
    
    # 加载测试数据
    # path = f'./datasets/{data_name}/{data_name}-dataset_test.txt'
    path = '/home/zh/cxy/code/IDCE_main/datasets/ml-1m/LightGCN_ml-1m-dataset_test.txt'
    test_df = pd.read_csv(path, header=None)
    test_user_list = defaultdict(list)
    for i, x in test_df.iterrows():
        test_user_list[x[0]].append(x[1])
        
    print(f"加载完成: 评分矩阵大小为 {scores_tensor_df.shape}, 测试集中用户数量为 {len(test_user_list)}")
    
    return scores_tensor_df, test_user_list

class IDCR:
    """物品动态中心性推荐框架的实现"""
    
    def __init__(self, scores_tensor_df, alpha=ALPHA, beta=BETA, gamma=GAMMA, 
                 lambda_param=LAMBDA, base_radius=BASE_RADIUS,
                 item_limit_num=ITEM_LIMIT_NUM, topk=TOPK, max_iter=MAX_ITER):
        """
        初始化IDCR模型
        
        Args:
            scores_tensor_df: 预训练的评分矩阵，形状为 [用户数, 物品数]
            alpha, beta, gamma: 中心性计算的权重参数
            lambda_param: 影响半径的缩放参数
            base_radius: 基础影响半径
            item_limit_num: 每个物品最多选择的用户数
            topk: 推荐的物品数量
            max_iter: 最大迭代次数
        """
        self.scores_df = scores_tensor_df
        self.scores = torch.tensor(scores_tensor_df.to_numpy())
        self.n_users, self.n_items = self.scores.shape
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_param = lambda_param
        self.base_radius = base_radius
        self.item_limit_num = item_limit_num
        self.topk = topk
        self.max_iter = max_iter
        
        print("预处理评分数据...")
        self._preprocess_scores()
        
        self.item_centrality = None
        self.item_radius = None
        
        # 矩阵缓存
        self.distance_matrix_cache = None
        
        # 物品相似度矩阵缓存
        self.item_similarity_cache = None
        
        print("IDCR模型初始化完成")
    
    def _preprocess_scores(self):
        """预处理评分矩阵，处理缺失值等"""
        self.scores[self.scores < -1e7] = 0
        # 排名
        self.scores_rank = torch.zeros_like(self.scores)
        for u in range(self.n_users):
            user_scores = self.scores[u, :]
            _, indices = torch.sort(user_scores, descending=True)
            ranks = torch.zeros_like(user_scores)
            ranks[indices] = torch.arange(self.n_items, dtype=ranks.dtype)
            self.scores_rank[u, :] = ranks
        nonzero_mask = self.scores > 0
        self.nonzero_users, self.nonzero_items = nonzero_mask.nonzero(as_tuple=True)
        print(f"有效的用户-物品交互数: {len(self.nonzero_users)}")
    
    def calculate_item_centrality(self):
        """计算物品中心性"""
        print("计算物品中心性指标...")
        scores = self.scores.numpy()
        n_users, n_items = scores.shape
        
        # 1. 计算度中心性 - 有多少用户对该物品有评分
        item_degree = np.zeros(n_items)
        for i in range(n_items):
            item_degree[i] = np.sum(scores[:, i] > 0)
        # 归一化度中心性
        max_degree = np.max(item_degree) if np.max(item_degree) > 0 else 1
        item_degree_centrality = item_degree / max_degree
        
        # 2. 特征中心性 - 使用物品流行度和平均评分
        item_avg_scores = np.zeros(n_items)
        for i in range(n_items):
            positive_scores = scores[:, i][scores[:, i] > 0]
            item_avg_scores[i] = np.mean(positive_scores) if len(positive_scores) > 0 else 0
        # 归一化特征中心性
        max_avg_score = np.max(item_avg_scores) if np.max(item_avg_scores) > 0 else 1
        item_feature_centrality = item_avg_scores / max_avg_score
        
        # 3. 介数中心性：使用物品相似度矩阵估计
        scores_sparse = sp.csr_matrix(scores)
        scores_sparse_T = scores_sparse.T
        print("计算物品相似度（分块）...")
        # 设置块大小 - 较大的块会使用更多内存但计算更快
        block_size = 200 
        item_similarity = np.zeros((n_items, n_items))
        # 分块计算相似度
        for i_start in tqdm(range(0, n_items, block_size), desc="计算物品相似度块"):
            i_end = min(i_start + block_size, n_items)
            # 获取物品块的用户评分
            items_i_data = scores_sparse_T[i_start:i_end].toarray()
            for j_start in range(0, n_items, block_size):
                j_end = min(j_start + block_size, n_items)
                if i_start > j_start and i_end > j_start:
                    continue
                items_j_data = scores_sparse_T[j_start:j_end].toarray()
                norms_i = np.sqrt(np.sum(items_i_data**2, axis=1))
                norms_j = np.sqrt(np.sum(items_j_data**2, axis=1))
                norms_i[norms_i == 0] = 1
                norms_j[norms_j == 0] = 1
                dot_products = np.dot(items_i_data, items_j_data.T)
                norms_matrix = np.outer(norms_i, norms_j)
                similarities = dot_products / norms_matrix
                item_similarity[i_start:i_end, j_start:j_end] = similarities
                if i_start != j_start:
                    item_similarity[j_start:j_end, i_start:i_end] = similarities.T
        
        # 缓存计算的相似度矩阵
        self.item_similarity_cache = item_similarity
                
        # 估计介数中心性：物品与其他物品的平均相似度越高，介数中心性越高
        item_betweenness_centrality = np.mean(item_similarity, axis=1)
        max_betweenness = np.max(item_betweenness_centrality) if np.max(item_betweenness_centrality) > 0 else 1
        item_betweenness_centrality = item_betweenness_centrality / max_betweenness
        
        # 4. 综合以上三种中心性，加权得到最终的物品中心性
        item_centrality = (self.alpha * item_degree_centrality + 
                           self.beta * item_betweenness_centrality + 
                           self.gamma * item_feature_centrality)
        max_centrality = np.max(item_centrality) if np.max(item_centrality) > 0 else 1
        item_centrality = item_centrality / max_centrality
        self.item_centrality = item_centrality
        print("物品中心性计算完成")
        return item_centrality
    
    def calculate_item_radius(self):
        """计算物品影响半径"""
        if self.item_centrality is None:
            self.calculate_item_centrality()
        self.item_radius = self.base_radius * (1 + self.lambda_param * self.item_centrality)
        return self.item_radius
    
    def calculate_distance_matrix(self):
        """计算用户-物品距离矩阵"""
        if self.distance_matrix_cache is not None:
            return self.distance_matrix_cache
             
        print("计算用户-物品距离矩阵...")
        distance_matrix = np.zeros((self.n_users, self.n_items))
        scores = self.scores.numpy()
        scores_min = np.min(scores[scores > -1e7])
        scores_max = np.max(scores)
        score_range = scores_max - scores_min
        
        if score_range > 0:
            for u in tqdm(range(self.n_users), desc="计算距离矩阵"):
                for i in range(self.n_items):
                    if scores[u, i] > -1e7:  
                        distance_matrix[u, i] = 1 - (scores[u, i] - scores_min) / score_range
                    else:
                        distance_matrix[u, i] = 1
        else:
            distance_matrix[:] = 0.5
            distance_matrix[scores <= -1e7] = 1

        self.distance_matrix_cache = distance_matrix
        return distance_matrix
    
    def calculate_item_user_probability(self):
        """计算物品选择用户的概率"""
        distance_matrix = self.calculate_distance_matrix()
        probability_matrix = np.zeros((self.n_items, self.n_users))
        for i in tqdm(range(self.n_items), desc="计算物品选择用户概率"):
            radius = self.item_radius[i]
            distance_weights = np.exp(-(distance_matrix[:, i]**2) / (radius**2))
            score_weights = self.scores[:, i].numpy()
            score_weights[score_weights < 0] = 0
            # 组合距离权重和分数权重
            combined_weights = distance_weights * score_weights
            if np.sum(combined_weights) > 0:
                probability_matrix[i, :] = combined_weights / np.sum(combined_weights)
            
        return probability_matrix
    
    def calculate_interest_transfer(self):
        """计算物品间的兴趣转移系数"""
        print("计算物品间的兴趣转移系数...")
        
        if self.item_centrality is None:
            self.calculate_item_centrality()

        if self.item_similarity_cache is not None:
            print("使用之前计算的缓存物品相似度矩阵...")
            item_similarity = self.item_similarity_cache
        else:
            # 如果缓存不可用，，，，，
            print("缓存不可用，重新计算物品相似度矩阵...")
            item_features = self.scores.T.numpy()
            item_similarity = cosine_similarity(item_features)
                
        transfer_matrix = np.zeros((self.n_items, self.n_items))
        for i in tqdm(range(self.n_items), desc="计算兴趣转移系数"):
            similarities = item_similarity[i, :]
            similarities[similarities < 0] = 0
            
            # 考虑中心性的影响 - 中心性高的物品更容易接收兴趣转移
            centrality_weighted_sim = similarities * self.item_centrality
            
            # 归一化得到转移系数
            if np.sum(centrality_weighted_sim) > 0:
                transfer_matrix[i, :] = centrality_weighted_sim / np.sum(centrality_weighted_sim)
        
        return transfer_matrix
    
    def check_convergence(self, current_selection, previous_selection, iteration, user_item_scores):
        """
        收敛判断方法，考虑排名变化和物品重要性
        Args:
            current_selection: 当前迭代的物品-用户选择矩阵
            previous_selection: 上一次迭代的物品-用户选择矩阵
            iteration: 当前迭代次数
            user_item_scores: 当前迭代计算的用户-物品分数
        Returns:
            bool: 是否已收敛
        """
        # 1. 基本变化率
        basic_change_ratio = np.sum(np.abs(current_selection - previous_selection)) / max(1, np.sum(previous_selection))
        
        # 2. 考虑排名变化的指标
        rank_stability = 0.0
        num_valid_users = 0
        
        for u in range(self.n_users):
            # 获取上一轮和当前轮对用户u有兴趣的物品
            prev_items = np.where(previous_selection[:, u] > 0)[0]
            curr_items = np.where(current_selection[:, u] > 0)[0]
            if len(prev_items) > 0 and len(curr_items) > 0:
                # 计算前K个物品的重叠度（推荐稳定性）
                k = min(self.topk, len(prev_items), len(curr_items))
                # 使用当前迭代计算的分数
                prev_scores = [(i, user_item_scores[u, i]) for i in prev_items]
                curr_scores = [(i, user_item_scores[u, i]) for i in curr_items]
                # 按评分排序
                prev_scores.sort(key=lambda x: x[1], reverse=True)
                curr_scores.sort(key=lambda x: x[1], reverse=True)
                # 获取排序后的前K个物品ID
                prev_top_k = set([i for i, _ in prev_scores[:k]])
                curr_top_k = set([i for i, _ in curr_scores[:k]])
                # 计算Jaccard相似度 (交集大小/并集大小)
                overlap = len(prev_top_k.intersection(curr_top_k))
                union = len(prev_top_k.union(curr_top_k))
                user_stability = overlap / max(1, union)
                rank_stability += user_stability
                num_valid_users += 1
        
        avg_rank_stability = rank_stability / max(1, num_valid_users)

        # 3. 考虑物品中心性的加权变化率
        weighted_change = 0.0
        total_centrality = np.sum(self.item_centrality)
        for i in range(self.n_items):
            item_change = np.sum(np.abs(current_selection[i, :] - previous_selection[i, :]))
            weighted_change += item_change * self.item_centrality[i]
        
        # 归一化加权变化率
        weighted_change_ratio = weighted_change / max(1e-10, total_centrality * self.n_users)
        
        # 4. 动态阈值：随着迭代次数增加，要求更高的稳定性，这些参数需要调整。。。。。
        base_threshold = 0.07  # 提高基本变化率阈值
        rank_threshold = 0.65  # 降低排名稳定性阈值
        weighted_threshold = 0.05  # 提高加权变化率阈值
        
        iteration_factor = min(0.6, 0.1 * iteration)
        dynamic_basic_threshold = base_threshold * (1 - iteration_factor)
        dynamic_rank_threshold = rank_threshold + (0.9 - rank_threshold) * iteration_factor
        dynamic_weighted_threshold = weighted_threshold * (1 - iteration_factor)
        
        # 5. 增加调试信息
        print(f"基本变化率: {basic_change_ratio:.4f} (阈值: {dynamic_basic_threshold:.4f})")
        print(f"排名稳定性: {avg_rank_stability:.4f} (阈值: {dynamic_rank_threshold:.4f})")
        print(f"加权变化率: {weighted_change_ratio:.4f} (阈值: {dynamic_weighted_threshold:.4f})")
        
        # 6. 调试信息
        total_selected_before = np.sum(previous_selection)
        total_selected_after = np.sum(current_selection)
        print(f"选择总数: 之前={total_selected_before}, 之后={total_selected_after}")
        
        # 显示前3个用户的推荐变化
        for u in range(min(3, self.n_users)):
            prev_items = np.where(previous_selection[:, u] > 0)[0]
            curr_items = np.where(current_selection[:, u] > 0)[0]
            added = set(curr_items) - set(prev_items)
            removed = set(prev_items) - set(curr_items)
            if len(added) > 0 or len(removed) > 0:
                print(f"用户{u}的推荐变化: 增加={added}, 移除={removed}")
        
        # 7. 综合判断条件
        is_converged = (
            (basic_change_ratio < dynamic_basic_threshold) and
            (avg_rank_stability > dynamic_rank_threshold) and
            (weighted_change_ratio < dynamic_weighted_threshold)
        )
        
        if is_converged:
            print(f"迭代收敛，提前结束于第{iteration+1}次迭代")
        
        return is_converged

    def _calculate_user_item_scores(self, item_user_selection, interest_transfer):
        """增强的用户-物品分数计算"""
        print("计算用户-物品分数...")
        user_item_scores = np.zeros((self.n_users, self.n_items))
        
        INTEREST_TRANSFER_BOOST = 1.5  # 兴趣转移增强系数
        for u in tqdm(range(self.n_users), desc="用户-物品分数计算"):
            selected_by_items = np.where(item_user_selection[:, u] > 0)[0]
            if len(selected_by_items) > 0:
                for i in selected_by_items:
                    # 物品i选择用户u的基础分数
                    base_score = self.scores[u, i].item()
                    # 通过兴趣转移影响其他物品的分数，采用非线性的兴趣转移增强
                    top_related_items = np.argsort(interest_transfer[i, :])[::-1][:50]  # 只考虑前50个相关物品
                    for j in top_related_items:
                        if j != i:
                            # 增强兴趣转移效果
                            transfer_coef = interest_transfer[i, j]
                            # 应用非线性转换增强小的转移系数
                            boosted_transfer = transfer_coef * INTEREST_TRANSFER_BOOST * (1 + np.log1p(transfer_coef))
                            transfer_score = base_score * boosted_transfer
                            user_item_scores[u, j] += transfer_score
                    # 物品自身的分数
                    user_item_scores[u, i] += base_score
            else:
                # 如果没有物品选择该用户，则使用原始分数，但添加随机扰动以增加变化性
                scores = self.scores[u, :].numpy()
                noise = np.random.normal(0, 0.01, size=self.n_items)
                user_item_scores[u, :] = scores + noise * (scores > 0)  # 只对正评分添加扰动
        return user_item_scores

    def recommend(self):
        """生成推荐结果"""
        print("开始生成IDCR推荐...")
        if self.item_centrality is None:
            self.calculate_item_centrality()
            self.calculate_item_radius()
        distance_matrix = self.calculate_distance_matrix()
        item_user_probability = self.calculate_item_user_probability()
        interest_transfer = self.calculate_interest_transfer()
        
        item_selected_users = {}
        
        # 对每个物品，基于概率选择用户
        print("初始化物品选择用户...")
        for i in range(self.n_items):
            user_probs = item_user_probability[i, :]
            top_users = np.argsort(user_probs)[::-1][:self.item_limit_num]
            item_selected_users[i] = top_users
        
        # 构建物品-用户选择矩阵
        item_user_selection = np.zeros((self.n_items, self.n_users))
        for i, users in item_selected_users.items():
            item_user_selection[i, users] = 1
        
        # 调试信息
        total_initial_selections = np.sum(item_user_selection)
        unique_items_selected = np.sum(np.sum(item_user_selection, axis=1) > 0)
        unique_users_selected = np.sum(np.sum(item_user_selection, axis=0) > 0)
        print(f"初始选择统计: 总选择数={total_initial_selections}, 被选中的物品数={unique_items_selected}, 被选中的用户数={unique_users_selected}")
        # 显示初始时前5个用户被哪些物品选择
        for u in range(min(5, self.n_users)):
            selected_by_items = np.where(item_user_selection[:, u] > 0)[0]
            if len(selected_by_items) > 0:
                print(f"用户{u}初始被这些物品选择: {selected_by_items[:10]}...")
        
        # 副本，用于跟踪迭代
        prev_selection = item_user_selection.copy()
        prev_user_item_scores = None
        
        # 双向选择迭代优化
        for iteration in range(self.max_iter):
            print(f"\n{'='*30}\n双向选择迭代: {iteration + 1}/{self.max_iter}\n{'='*30}")
            
            # 1. 用户选择物品 - 计算用户-物品分数
            user_item_scores = self._calculate_user_item_scores(item_user_selection, interest_transfer)
            # 如果不是第一次迭代，输出分数变化
            if prev_user_item_scores is not None:
                score_diff = np.abs(user_item_scores - prev_user_item_scores)
                avg_score_change = np.mean(score_diff[score_diff > 0])
                max_score_change = np.max(score_diff)
                print(f"分数变化统计: 平均变化={avg_score_change:.4f}, 最大变化={max_score_change:.4f}")
            
            # 2. 物品重新选择用户
            print("重新选择用户...")
            new_item_user_selection = np.zeros((self.n_items, self.n_users))
            for i in tqdm(range(self.n_items), desc="物品选择用户"):
                # 根据用户-物品分数和物品影响半径重新计算选择概率
                item_scores = user_item_scores[:, i]
                # 结合距离权重
                radius = self.item_radius[i]
                distance_weights = np.exp(-(distance_matrix[:, i]**2) / (radius**2))
                # 随机扰动以避免陷入局部最优
                if iteration > 0:
                    noise = np.random.normal(0, 0.05, size=self.n_users) * (item_scores > 0)
                    combined_weights = (item_scores + noise) * distance_weights
                else:
                    combined_weights = item_scores * distance_weights
                # 选择权重最高的用户
                if np.sum(combined_weights) > 0:
                    top_users = np.argsort(combined_weights)[::-1][:self.item_limit_num]
                    new_item_user_selection[i, top_users] = 1
            item_user_selection = new_item_user_selection
            
            # 调试信息
            total_selected_before = np.sum(prev_selection)
            total_selected_after = np.sum(item_user_selection)
            diff_count = np.sum(np.abs(item_user_selection - prev_selection))
            changed_percent = diff_count / total_selected_before * 100 if total_selected_before > 0 else 0
            items_with_users = np.sum(np.sum(item_user_selection, axis=1) > 0)
            users_with_items = np.sum(np.sum(item_user_selection, axis=0) > 0)
            print(f"\n迭代{iteration+1}变化统计:")
            print(f"总变化数={diff_count}, 变化比例={changed_percent:.2f}%")
            print(f"选择总数: 之前={total_selected_before}, 之后={total_selected_after}")
            print(f"选择的物品数={items_with_users}, 选择的用户数={users_with_items}")
            for u in range(min(5, self.n_users)):
                prev_items = np.where(prev_selection[:, u] > 0)[0]
                curr_items = np.where(item_user_selection[:, u] > 0)[0]
                added = set(curr_items) - set(prev_items)
                removed = set(prev_items) - set(curr_items)
                if len(added) > 0 or len(removed) > 0:
                    print(f"用户{u}的推荐变化: 增加={added}, 移除={removed}")
            
            if iteration > 0: 
                is_converged = self.check_convergence(item_user_selection, prev_selection, iteration, user_item_scores)
                if is_converged:
                    break
            
            prev_selection = item_user_selection.copy()
            prev_user_item_scores = user_item_scores.copy()
        
        print(f"\n{'='*30}\n迭代完成，生成最终推荐列表\n{'='*30}")
        print("生成最终推荐列表...")
        recommendations = {}
        for u in tqdm(range(self.n_users), desc="生成用户推荐列表"):
            selected_items = np.where(item_user_selection[:, u] > 0)[0]
            if len(selected_items) < self.topk:
                original_scores = self.scores[u, :].numpy()
                for i in selected_items:
                    original_scores[i] = -float('inf')
                # 选择评分最高的物品补充
                additional_items = np.argsort(original_scores)[::-1][:(self.topk - len(selected_items))]
                recommended_items = np.concatenate([selected_items, additional_items])
            else:
                # 如果超过topk，选择评分最高的topk个
                item_scores = [(i, self.scores[u, i].item()) for i in selected_items]
                item_scores.sort(key=lambda x: x[1], reverse=True)
                recommended_items = np.array([i for i, _ in item_scores[:self.topk]])
            recommendations[u] = recommended_items
        return recommendations
    
    def calculate_dcb(self, recommendations, test_user_list=None):
        """
        计算动态覆盖平衡度 (Dynamic Coverage Balance, DCB)
        Args:
            recommendations: 推荐结果，字典
            test_user_list: 测试集，每个用户对应的物品列表
        Returns:
            dcb: 动态覆盖平衡度
        """
        if self.item_centrality is None:
            self.calculate_item_centrality()
        
        # 推荐频率
        recommendation_frequency = np.zeros(self.n_items)
        for u, items in recommendations.items():
            for i in items:
                recommendation_frequency[i] += 1
        
        # 计算推荐频率的基尼系数
        sorted_freq = np.sort(recommendation_frequency)
        n = len(sorted_freq)
        cum_freq = np.cumsum(sorted_freq)
        total_freq = cum_freq[-1]
        # 计算洛伦兹曲线下的面积
        area = np.sum(cum_freq) / total_freq / n
        # 基尼系数 = 1 - 2 * 面积
        gini = 1 - 2 * area
        
        # 计算推荐频率与中心性的比例差异
        if total_freq > 0:
            freq_ratio = recommendation_frequency / total_freq
        else:
            freq_ratio = np.zeros_like(recommendation_frequency)
        
        centrality_sum = np.sum(self.item_centrality)
        if centrality_sum > 0:
            centrality_ratio = self.item_centrality / centrality_sum
        else:
            centrality_ratio = np.ones_like(self.item_centrality) / self.n_items
        # 计算比例差异的平均绝对值
        ratio_diff_mean = np.mean(np.abs(freq_ratio - centrality_ratio))
        # 计算DCB
        dcb = (1 - gini) * (1 - ratio_diff_mean)
        print(f"推荐频率的基尼系数: {gini:.4f}")
        print(f"比例差异的平均绝对值: {ratio_diff_mean:.4f}")
        print(f"动态覆盖平衡度 (DCB): {dcb:.4f}")
        return dcb
    
    def evaluate(self, recommendations, test_user_list):
        """
        评估推荐结果
        Args:
            recommendations: 推荐结果，字典
            test_user_list: 测试集，每个用户对应的物品列表
        Returns:
            metrics: 评估指标字典
        """
        print("开始评估推荐结果...")
        users = list(recommendations.keys())
        n_users = len(users)
        precision_list = []
        recall_list = []
        ndcg_list = []
        
        for u in users:
            rec_items = recommendations[u]
            test_items = set(test_user_list[u])
            # 精确率
            hits = len(set(rec_items) & test_items)
            precision = hits / len(rec_items) if len(rec_items) > 0 else 0
            precision_list.append(precision)
            # 召回率
            recall = hits / len(test_items) if len(test_items) > 0 else 0
            recall_list.append(recall)
            # NDCG
            ndcg = 0
            for i, item in enumerate(rec_items):
                if item in test_items:
                    ndcg += 1 / np.log2(i + 2)  # i+2 因为 log2(1) = 0
            # 理想DCG
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(test_items), len(rec_items))))
            ndcg = ndcg / idcg if idcg > 0 else 0
            ndcg_list.append(ndcg)
        # 平均值
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_ndcg = np.mean(ndcg_list)
        # 覆盖率
        all_rec_items = set()
        for items in recommendations.values():
            all_rec_items.update(items)
        coverage = len(all_rec_items) / self.n_items
        # 动态覆盖平衡度
        dcb = self.calculate_dcb(recommendations, test_user_list)
        
        metrics = {
            'precision': avg_precision,
            'recall': avg_recall,
            'ndcg': avg_ndcg,
            'coverage': coverage,
            'dcb': dcb
        }
        print("评估指标:")
        print(f"精确率: {avg_precision:.4f}")
        print(f"召回率: {avg_recall:.4f}")
        print(f"NDCG: {avg_ndcg:.4f}")
        print(f"覆盖率: {coverage:.4f}")
        print(f"动态覆盖平衡度 (DCB): {dcb:.4f}")
        return metrics


def run_baseline(scores_tensor_df, test_user_list, topk=10):
    """
    运行基准方法，直接使用预训练评分进行推荐
    Args:
        scores_tensor_df: 预训练的评分矩阵
        test_user_list: 测试集，每个用户对应的物品列表
        topk: 推荐的物品数量
    Returns:
        recommendations: 推荐结果
    """
    print("运行基准方法...")
    scores = torch.tensor(scores_tensor_df.to_numpy())
    n_users, n_items = scores.shape
    scores[scores < -1e7] = float('-inf')
    recommendations = {}
    for u in range(n_users):
        user_scores = scores[u, :].numpy()
        top_items = np.argsort(user_scores)[::-1][:topk]
        recommendations[u] = top_items
    
    n_users = len(recommendations)
    precision_list = []
    recall_list = []
    ndcg_list = []
    
    for u in recommendations.keys():
        rec_items = recommendations[u]
        test_items = set(test_user_list[u])
        hits = len(set(rec_items) & test_items)
        precision = hits / len(rec_items) if len(rec_items) > 0 else 0
        precision_list.append(precision)
        recall = hits / len(test_items) if len(test_items) > 0 else 0
        recall_list.append(recall)
        ndcg = 0
        for i, item in enumerate(rec_items):
            if item in test_items:
                ndcg += 1 / np.log2(i + 2)  # i+2 因为 log2(1) = 0
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(test_items), len(rec_items))))
        ndcg = ndcg / idcg if idcg > 0 else 0
        ndcg_list.append(ndcg)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_ndcg = np.mean(ndcg_list)
    all_rec_items = set()
    for items in recommendations.values():
        all_rec_items.update(items)
    coverage = len(all_rec_items) / n_items
    
    # 打印评估结果
    print("基准方法评估指标:")
    print(f"精确率: {avg_precision:.4f}")
    print(f"召回率: {avg_recall:.4f}")
    print(f"NDCG: {avg_ndcg:.4f}")
    print(f"覆盖率: {coverage:.4f}")
    
    return recommendations


def main():
    """主函数"""
    print("=" * 50)
    print("物品动态中心性推荐框架 (IDCR) 运行")
    print("=" * 50)
    
    try:
        scores_tensor_df, test_user_list = load_data()
        all_metrics = {}
        # 基准测试运行
        print("\n" + "=" * 30)
        print("基准方法评估")
        print("=" * 30)
        baseline_recommendations = run_baseline(scores_tensor_df, test_user_list)
        baseline_metrics = evaluate_recommendations(baseline_recommendations, test_user_list, scores_tensor_df.shape[1])
        all_metrics['基准方法'] = baseline_metrics
        # 运行IDCR方法
        print("\n" + "=" * 30)
        print("IDCR方法评估")
        print("=" * 30)
        idcr = IDCR(scores_tensor_df)
        idcr_recommendations = idcr.recommend()
        idcr_metrics = idcr.evaluate(idcr_recommendations, test_user_list)
        all_metrics['IDCR方法'] = idcr_metrics
        # 结果对比
        print("\n" + "=" * 50)
        print("结果对比")
        print("=" * 50)
        print("\n基准方法DCB评估:")
        baseline_dcb = idcr.calculate_dcb(baseline_recommendations, test_user_list)
        all_metrics['基准方法']['dcb'] = baseline_dcb

        print("\n" + "=" * 70)
        print(f"{'方法名称':<15}{'精确率':<10}{'召回率':<10}{'NDCG':<10}{'覆盖率':<10}{'DCB':<10}")
        print("-" * 70)        
        for method_name, metrics in all_metrics.items():
            print(f"{method_name:<15}{metrics['precision']:.4f}{' '*6}{metrics['recall']:.4f}{' '*6}{metrics['ndcg']:.4f}{' '*6}{metrics['coverage']:.4f}{' '*6}{metrics['dcb']:.4f}")
        
        print("=" * 70)
        
        # 分析指标变化
        baseline_metrics = all_metrics['基准方法']
        idcr_metrics = all_metrics['IDCR方法']
        prec_change = (idcr_metrics['precision'] - baseline_metrics['precision']) / baseline_metrics['precision'] * 100
        recall_change = (idcr_metrics['recall'] - baseline_metrics['recall']) / baseline_metrics['recall'] * 100
        ndcg_change = (idcr_metrics['ndcg'] - baseline_metrics['ndcg']) / baseline_metrics['ndcg'] * 100
        cov_change = (idcr_metrics['coverage'] - baseline_metrics['coverage']) / baseline_metrics['coverage'] * 100
        dcb_change = (idcr_metrics['dcb'] - baseline_metrics['dcb']) / baseline_metrics['dcb'] * 100
        print("\nIDCR相比基准方法的指标变化:")
        print(f"精确率: {prec_change:.2f}%, 召回率: {recall_change:.2f}%, NDCG: {ndcg_change:.2f}%")
        print(f"覆盖率: {cov_change:.2f}%, DCB: {dcb_change:.2f}%")
        
    except Exception as e:
        import traceback
        print(f"运行过程中发生错误: {e}")
        traceback.print_exc()
        print("byebye")

#通用的评估函数，用于基准方法
def evaluate_recommendations(recommendations, test_user_list, n_items):
    """
    评估推荐结果，返回评估指标
    Args:
        recommendations: 推荐结果，字典
        test_user_list: 测试数据集
        n_items: 物品总数       
    Returns:
        metrics: 包含各项评估指标的字典
    """
    n_users = len(recommendations)
    precision_list = []
    recall_list = []
    ndcg_list = []
    for u in recommendations.keys():
        rec_items = recommendations[u]
        test_items = set(test_user_list[u])
        hits = len(set(rec_items) & test_items)
        precision = hits / len(rec_items) if len(rec_items) > 0 else 0
        precision_list.append(precision)
        recall = hits / len(test_items) if len(test_items) > 0 else 0
        recall_list.append(recall)
        ndcg = 0
        for i, item in enumerate(rec_items):
            if item in test_items:
                ndcg += 1 / np.log2(i + 2) 
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(test_items), len(rec_items))))
        ndcg = ndcg / idcg if idcg > 0 else 0
        ndcg_list.append(ndcg)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_ndcg = np.mean(ndcg_list)
    all_rec_items = set()
    for items in recommendations.values():
        all_rec_items.update(items)
    coverage = len(all_rec_items) / n_items
    metrics = {
        'precision': avg_precision,
        'recall': avg_recall,
        'ndcg': avg_ndcg,
        'coverage': coverage,
        'dcb': None  # DCB将在主函数中使用IDCR模型计算
    }
    return metrics


if __name__ == "__main__":
    main() 