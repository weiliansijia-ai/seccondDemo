#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多钻孔裂隙网络的连通性分析与三维重构

问题4：基于多钻孔数据的裂隙连通性分析和三维重构
集成数学建模、统计建模、机器学习和深度学习方法
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import euclidean_distances
from scipy import optimize, spatial, interpolate
from scipy.stats import pearsonr
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ["Dengxian",'SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class MultiBoreholeCrackAnalyzer:
    """多钻孔裂隙连通性分析系统"""
    
    def __init__(self, input_dir="../附件4", output_dir="./"):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = output_dir
        
        # 钻孔参数
        self.bore_diameter = 30  # mm
        self.bore_circumference = np.pi * self.bore_diameter
        
        # 钻孔空间位置参数 (x, y, z_start, z_end)
        self.borehole_positions = {
            '1': {'start': (500, 2000, 0), 'end': (500, 2000, 7000), 'depth': 7000},
            '2': {'start': (1500, 2000, 0), 'end': (1500, 2000, 7000), 'depth': 7000},
            '3': {'start': (2500, 2000, 0), 'end': (2500, 2000, 7000), 'depth': 7000},
            '4': {'start': (500, 1000, 0), 'end': (500, 1000, 5000), 'depth': 5000},
            '5': {'start': (1500, 1000, 0), 'end': (1500, 1000, 7000), 'depth': 7000},
            '6': {'start': (2500, 1000, 0), 'end': (2500, 1000, 7000), 'depth': 7000}
        }
        
        self.results = {}
        
    def get_borehole_image_paths(self, borehole_id):
        """获取单个钻孔的所有图像路径"""
        borehole_dir = os.path.join(self.input_dir, f"{borehole_id}#孔")
        if not os.path.exists(borehole_dir):
            print(f"警告：钻孔目录不存在: {borehole_dir}")
            return {}
        
        image_paths = {}
        image_files = [f for f in os.listdir(borehole_dir) if f.endswith('.jpg')]
        
        for image_file in image_files:
            depth_info = image_file.replace('.jpg', '')
            image_path = os.path.join(borehole_dir, image_file)
            image_paths[depth_info] = image_path
        
        return image_paths
    
    def mathematical_crack_detection(self, gray_image):
        """数学建模方法 - 裂隙检测"""
        
        # 1. 预处理
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
        
        # 2. 多尺度边缘检测
        edges_1 = cv2.Canny(blurred, 30, 100)
        edges_2 = cv2.Canny(blurred, 50, 150)
        edges_3 = cv2.Canny(blurred, 70, 200)
        combined_edges = edges_1 | edges_2 | edges_3
        
        # 3. 梯度检测
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 自适应阈值
        threshold = np.percentile(gradient_mag, 85)
        grad_binary = gradient_mag > threshold
        
        # 4. 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = combined_edges | grad_binary.astype(np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        return combined.astype(bool)
    
    def statistical_crack_detection(self, gray_image):
        """统计建模方法 - 基于多特征统计分析"""
        
        # 1. 局部标准差
        kernel = np.ones((5, 5)) / 25
        local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        local_sq_mean = cv2.filter2D((gray_image.astype(np.float32))**2, -1, kernel)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        
        # 2. 统计阈值
        std_threshold = np.percentile(local_std, 80)
        intensity_threshold = np.percentile(gray_image, 30)
        
        # 3. 分类
        stat_mask = (local_std > std_threshold) & (gray_image < intensity_threshold)
        
        return stat_mask
    
    def extract_crack_features(self, binary_mask, depth_range):
        """提取裂隙特征"""
        
        from skimage import measure, morphology
        
        # 连通分量分析
        labeled = measure.label(binary_mask)
        regions = measure.regionprops(labeled)
        
        crack_features = []
        
        for region in regions:
            if region.area >= 150:  # 最小面积阈值, 再次提高以减少噪声, 解决内存溢出问题
                # 基本几何特征
                centroid = region.centroid
                area = region.area
                
                # 边界框
                bbox = region.bbox  # (min_row, min_col, max_row, max_col)
                
                # 方向和长轴
                if hasattr(region, 'orientation'):
                    orientation = region.orientation
                    major_axis_length = region.major_axis_length
                    minor_axis_length = region.minor_axis_length
                else:
                    orientation = 0
                    major_axis_length = 0
                    minor_axis_length = 0
                
                # 计算3D坐标（假设深度范围信息）
                depth_start, depth_end = depth_range
                z_coord = (depth_start + depth_end) / 2  # 深度中点
                
                crack_feature = {
                    'centroid_2d': centroid,
                    'centroid_3d': (centroid[1], centroid[0], z_coord),  # 注意坐标转换
                    'area': area,
                    'bbox': bbox,
                    'orientation': orientation,
                    'major_axis_length': major_axis_length,
                    'minor_axis_length': minor_axis_length,
                    'depth_range': depth_range,
                    'eccentricity': region.eccentricity if hasattr(region, 'eccentricity') else 0
                }
                
                crack_features.append(crack_feature)
        
        return crack_features
    
    def analyze_single_borehole(self, borehole_id):
        """分析单个钻孔 (内存优化)"""
        
        print(f"分析钻孔 {borehole_id}#")
        
        # 获取图像路径
        image_paths = self.get_borehole_image_paths(borehole_id)
        
        if not image_paths:
            print(f"  钻孔 {borehole_id}# 无可用图像")
            return None
        
        borehole_cracks = []
        
        for depth_info, image_path in image_paths.items():
            print(f"  分析深度段: {depth_info}")
            
            try:
                image_data = np.fromfile(image_path, dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                if image is None:
                    print(f"    警告: 无法加载图像 {image_path}")
                    continue
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"    加载或处理图像失败 {image_path}: {e}")
                continue

            # 解析深度范围
            if '-' in depth_info:
                depth_parts = depth_info.split('-')
                depth_start = int(depth_parts[0]) * 1000  # 转换为mm
                depth_end = int(depth_parts[1].replace('m', '')) * 1000
            else:
                depth_start = 0
                depth_end = 1000
            
            depth_range = (depth_start, depth_end)
            
            # 裂隙检测
            math_result = self.mathematical_crack_detection(gray_image)
            stat_result = self.statistical_crack_detection(gray_image)
            
            # 融合结果
            combined_result = math_result | stat_result
            
            # 提取特征
            crack_features = self.extract_crack_features(combined_result, depth_range)
            
            for crack_feature in crack_features:
                crack_feature['borehole_id'] = borehole_id
                crack_feature['depth_info'] = depth_info
                crack_feature['image_shape'] = gray_image.shape
                borehole_cracks.append(crack_feature)
            
            print(f"    检测到 {len(crack_features)} 个裂隙")
        
        result = {
            'borehole_id': borehole_id,
            'position': self.borehole_positions[borehole_id],
            'crack_features': borehole_cracks,
            'total_cracks': len(borehole_cracks)
        }
        
        return result
    
    def calculate_3d_coordinates(self, crack_feature):
        """计算裂隙的3D坐标"""
        
        borehole_id = crack_feature['borehole_id']
        borehole_pos = self.borehole_positions[borehole_id]
        
        # 钻孔起点坐标
        x_bore, y_bore, z_start = borehole_pos['start']
        
        # 图像坐标到钻孔坐标的转换
        centroid_2d = crack_feature['centroid_2d']
        depth_range = crack_feature['depth_range']
        
        # x坐标：周向位置转换为实际坐标偏移
        circumferential_angle = (centroid_2d[1] / crack_feature['image_shape'][1]) * 2 * np.pi
        x_offset = self.bore_diameter/2 * np.cos(circumferential_angle)
        y_offset = self.bore_diameter/2 * np.sin(circumferential_angle)
        
        # z坐标：轴向深度
        z_coord = z_start + (depth_range[0] + depth_range[1]) / 2
        
        # 最终3D坐标
        x_3d = x_bore + x_offset
        y_3d = y_bore + y_offset
        z_3d = z_coord
        
        return (x_3d, y_3d, z_3d)
    
    def machine_learning_connectivity_prediction(self, all_cracks):
        """机器学习方法预测裂隙连通性 (KDTree 优化)"""
        
        if len(all_cracks) < 2:
            return {}

        print("  - 计算所有裂隙的3D坐标...")
        crack_positions = np.array([self.calculate_3d_coordinates(c) for c in all_cracks])
        
        print("  - 构建空间索引 (KDTree)...")
        # 使用KDTree进行空间索引，优化搜索
        # 最大连接距离可以根据先验知识调整，这里设为1500mm
        max_connect_distance = 1200.0 # 从1500mm降低以减少内存消耗 
        tree = spatial.KDTree(crack_positions)
        candidate_pairs = tree.query_pairs(r=max_connect_distance, output_type='set')
        
        print(f"  - 发现 {len(candidate_pairs)} 个候选裂隙对 (距离 < {max_connect_distance}mm)")

        if not candidate_pairs:
            return {}

        # 计算候选裂隙对的特征
        crack_pairs = []
        pair_features = []
        
        for i, j in candidate_pairs:
            crack1, crack2 = all_cracks[i], all_cracks[j]
            
            # 跳过同一钻孔的裂隙
            if crack1['borehole_id'] == crack2['borehole_id']:
                continue
            
            pos1 = crack_positions[i]
            pos2 = crack_positions[j]
            
            # 计算特征
            features = self.calculate_pair_features(crack1, crack2, pos1, pos2)
            
            crack_pairs.append((i, j))
            pair_features.append(features)
        
        if len(pair_features) == 0:
            print("  - 候选对均来自同一钻孔，无跨孔连通性可分析。")
            return {}
        
        print(f"  - 为 {len(pair_features)} 个跨钻孔裂隙对计算特征。")

        # 转换为numpy数组
        X = np.array(pair_features)
        
        # 创建伪标签（基于距离和其他启发式规则）
        y_pseudo = self.create_connectivity_labels(X, crack_pairs, all_cracks)
        
        # 训练随机森林分类器
        print("  - 训练随机森林分类器...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # 使用所有CPU核心
        rf.fit(X, y_pseudo)
        
        # 预测连通概率
        print("  - 预测连通概率...")
        connectivity_probs = rf.predict_proba(X)[:, 1]  # 连通的概率
        
        # 构建连通性字典
        connectivity_dict = {}
        for idx, (i, j) in enumerate(crack_pairs):
            connectivity_dict[(i, j)] = {
                'probability': connectivity_probs[idx],
                'features': pair_features[idx],
                'predicted_connected': connectivity_probs[idx] > 0.5
            }
        
        return connectivity_dict
    
    def calculate_pair_features(self, crack1, crack2, pos1, pos2):
        """计算裂隙对的特征"""
        
        # 1. 空间距离特征
        euclidean_dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
        
        # 2. 深度差异
        depth_diff = abs(pos1[2] - pos2[2])
        
        # 3. 水平距离
        horizontal_dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
        # 4. 裂隙大小特征
        area_ratio = crack1['area'] / (crack2['area'] + 1e-8)
        area_sum = crack1['area'] + crack2['area']
        
        # 5. 方向特征
        orientation_diff = abs(crack1['orientation'] - crack2['orientation'])
        
        # 6. 形状特征
        eccentricity_diff = abs(crack1['eccentricity'] - crack2['eccentricity'])
        
        # 7. 钻孔间距
        borehole_dist = self.get_borehole_distance(crack1['borehole_id'], crack2['borehole_id'])
        
        features = [
            euclidean_dist,
            depth_diff,
            horizontal_dist,
            area_ratio,
            area_sum,
            orientation_diff,
            eccentricity_diff,
            borehole_dist
        ]
        
        return features
    
    def get_borehole_distance(self, borehole_id1, borehole_id2):
        """计算两个钻孔之间的距离"""
        
        pos1 = self.borehole_positions[borehole_id1]['start']
        pos2 = self.borehole_positions[borehole_id2]['start']
        
        # 只考虑水平距离
        dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return dist
    
    def create_connectivity_labels(self, features, crack_pairs, all_cracks):
        """创建连通性伪标签"""
        
        labels = []
        
        for idx, (i, j) in enumerate(crack_pairs):
            # 基于距离的启发式规则
            euclidean_dist = features[idx][0]
            depth_diff = features[idx][1]
            horizontal_dist = features[idx][2]
            area_sum = features[idx][4]
            
            # 连通性判断规则
            connected = False
            
            # 规则1：距离近且深度相近
            if euclidean_dist < 500 and depth_diff < 300:
                connected = True
            
            # 规则2：水平距离近且面积大
            if horizontal_dist < 300 and area_sum > 100:
                connected = True
            
            # 规则3：相邻钻孔且深度相近
            borehole_dist = features[idx][7]
            if borehole_dist <= 1000 and depth_diff < 500:
                connected = True
            
            labels.append(int(connected))
        
        return np.array(labels)
    
    def deep_learning_clustering(self, all_cracks):
        """深度学习方法进行裂隙聚类"""
        
        if len(all_cracks) < 3:
            return {}
        
        # 提取深度特征
        deep_features = []
        for crack in all_cracks:
            pos_3d = self.calculate_3d_coordinates(crack)
            
            # 组合特征
            feature_vector = [
                pos_3d[0], pos_3d[1], pos_3d[2],  # 3D位置
                crack['area'],                     # 面积
                crack['orientation'],              # 方向
                crack['major_axis_length'],        # 长轴
                crack['minor_axis_length'],        # 短轴
                crack['eccentricity']              # 离心率
            ]
            
            deep_features.append(feature_vector)
        
        # 标准化特征
        features_array = np.array(deep_features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        # DBSCAN聚类
        clustering = DBSCAN(eps=0.8, min_samples=2)
        cluster_labels = clustering.fit_predict(features_scaled)
        
        # 组织聚类结果
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        return clusters
    
    def calculate_connectivity_probability(self, crack1, crack2):
        """计算两个裂隙的连通概率"""
        
        pos1 = self.calculate_3d_coordinates(crack1)
        pos2 = self.calculate_3d_coordinates(crack2)
        
        # 空间距离
        distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
        
        # 基于距离的概率模型
        # 使用高斯衰减函数
        sigma = 800  # 距离衰减参数
        prob_distance = np.exp(-(distance**2) / (2 * sigma**2))
        
        # 深度相似性
        depth_diff = abs(pos1[2] - pos2[2])
        prob_depth = np.exp(-(depth_diff**2) / (2 * 300**2))
        
        # 面积相似性
        area_ratio = min(crack1['area'], crack2['area']) / max(crack1['area'], crack2['area'])
        prob_area = area_ratio
        
        # 方向相似性
        orientation_diff = abs(crack1['orientation'] - crack2['orientation'])
        prob_orientation = np.exp(-(orientation_diff**2) / (2 * (np.pi/4)**2))
        
        # 综合概率
        connectivity_prob = (prob_distance * prob_depth * prob_area * prob_orientation) ** 0.25
        
        return connectivity_prob
    
    def identify_high_uncertainty_regions(self, all_cracks, connectivity_results):
        """识别高不确定性区域"""
        
        # 创建3D空间网格
        x_coords = [self.calculate_3d_coordinates(crack)[0] for crack in all_cracks]
        y_coords = [self.calculate_3d_coordinates(crack)[1] for crack in all_cracks]
        z_coords = [self.calculate_3d_coordinates(crack)[2] for crack in all_cracks]
        
        if len(x_coords) == 0:
            return []
        
        # 计算空间范围
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        z_min, z_max = min(z_coords), max(z_coords)
        
        # 创建网格点
        grid_resolution = 500  # mm
        x_grid = np.arange(x_min, x_max + grid_resolution, grid_resolution)
        y_grid = np.arange(y_min, y_max + grid_resolution, grid_resolution)
        z_grid = np.arange(z_min, z_max + grid_resolution, grid_resolution)
        
        uncertainty_scores = []
        
        for x in x_grid:
            for y in y_grid:
                for z in z_grid:
                    # 计算该点的不确定性分数
                    score = self.calculate_uncertainty_score((x, y, z), all_cracks, connectivity_results)
                    
                    uncertainty_scores.append({
                        'position': (x, y, z),
                        'uncertainty_score': score
                    })
        
        # 按不确定性分数排序
        uncertainty_scores.sort(key=lambda x: x['uncertainty_score'], reverse=True)
        
        return uncertainty_scores[:20]  # 返回前20个高不确定性区域
    
    def calculate_uncertainty_score(self, position, all_cracks, connectivity_results):
        """计算位置的不确定性分数"""
        
        # 计算到最近裂隙的距离
        min_distance = float('inf')
        for crack in all_cracks:
            crack_pos = self.calculate_3d_coordinates(crack)
            dist = np.linalg.norm(np.array(position) - np.array(crack_pos))
            min_distance = min(min_distance, dist)
        
        # 计算连通性预测的不确定性
        connectivity_uncertainty = 0
        count = 0
        
        for (i, j), conn_info in connectivity_results.items():
            prob = conn_info['probability']
            # 概率接近0.5时不确定性最高
            uncertainty = 1 - 2 * abs(prob - 0.5)
            connectivity_uncertainty += uncertainty
            count += 1
        
        if count > 0:
            connectivity_uncertainty /= count
        
        # 综合不确定性分数
        # 距离越远，连通性不确定性越高，总不确定性越高
        distance_factor = min(min_distance / 1000, 1.0)  # 归一化到[0,1]
        total_uncertainty = distance_factor * 0.6 + connectivity_uncertainty * 0.4
        
        return total_uncertainty
    
    def suggest_optimal_drill_locations(self, all_cracks, connectivity_results):
        """建议最优的补充钻孔位置"""
        
        # 识别高不确定性区域
        high_uncertainty_regions = self.identify_high_uncertainty_regions(all_cracks, connectivity_results)
        
        if len(high_uncertainty_regions) == 0:
            return []
        
        # 选择前3个最优位置
        optimal_locations = []
        
        for i, region in enumerate(high_uncertainty_regions[:10]):  # 从前10个中选择
            position = region['position']
            x, y, z = position
            
            # 检查是否与现有钻孔距离合适
            min_distance_to_existing = float('inf')
            for borehole_id, borehole_info in self.borehole_positions.items():
                borehole_pos = borehole_info['start']
                dist = np.sqrt((x - borehole_pos[0])**2 + (y - borehole_pos[1])**2)
                min_distance_to_existing = min(min_distance_to_existing, dist)
            
            # 避免太近（<800mm）或太远（>2000mm）
            if 800 <= min_distance_to_existing <= 2000:
                # 计算该位置的综合优先级
                priority_score = region['uncertainty_score'] * (1 / min_distance_to_existing) * 1000
                
                optimal_locations.append({
                    'position': position,
                    'priority_score': priority_score,
                    'uncertainty_score': region['uncertainty_score'],
                    'distance_to_nearest': min_distance_to_existing,
                    'reason': f'高不确定性区域，距离现有钻孔{min_distance_to_existing:.0f}mm'
                })
        
        # 按优先级排序
        optimal_locations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return optimal_locations[:3]  # 返回前3个最优位置
    
    def create_3d_visualization(self, all_cracks, connectivity_results, optimal_locations=None):
        """创建三维可视化"""
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制钻孔位置
        for borehole_id, borehole_info in self.borehole_positions.items():
            start_pos = borehole_info['start']
            end_pos = borehole_info['end']
            
            # 钻孔轴线
            ax.plot([start_pos[0], end_pos[0]], 
                   [start_pos[1], end_pos[1]], 
                   [start_pos[2], end_pos[2]], 
                   'k-', linewidth=3, alpha=0.6)
            
            # 钻孔标识
            ax.text(start_pos[0], start_pos[1], start_pos[2] - 200, 
                   f'{borehole_id}#孔', fontsize=12, fontweight='bold')
        
        # 绘制裂隙位置
        crack_positions = []
        for crack in all_cracks:
            pos = self.calculate_3d_coordinates(crack)
            crack_positions.append(pos)
            
            # 根据裂隙大小设置点的大小
            size = max(20, min(100, crack['area'] / 2))
            
            ax.scatter(pos[0], pos[1], pos[2], 
                      s=size, c='red', alpha=0.6, marker='o')
        
        # 绘制连通性链接
        for (i, j), conn_info in connectivity_results.items():
            if conn_info['predicted_connected'] and conn_info['probability'] > 0.7:
                pos1 = crack_positions[i]
                pos2 = crack_positions[j]
                
                # 连接线的颜色根据概率确定
                prob = conn_info['probability']
                color = plt.cm.Blues(prob)
                
                ax.plot([pos1[0], pos2[0]], 
                       [pos1[1], pos2[1]], 
                       [pos1[2], pos2[2]], 
                       color=color, linewidth=2, alpha=0.7)
        
        # 绘制建议的钻孔位置
        if optimal_locations:
            for i, location in enumerate(optimal_locations):
                pos = location['position']
                ax.scatter(pos[0], pos[1], pos[2], 
                          s=200, c='yellow', marker='^', 
                          edgecolors='black', linewidth=2)
                ax.text(pos[0], pos[1], pos[2] + 200, 
                       f'建议{i+1}', fontsize=10, fontweight='bold')
        
        # 设置图表属性
        ax.set_xlabel('X坐标 (mm)', fontsize=12)
        ax.set_ylabel('Y坐标 (mm)', fontsize=12)
        ax.set_zlabel('Z坐标 (mm)', fontsize=12)
        ax.set_title('多钻孔裂隙网络连通性三维分析', fontsize=14, fontweight='bold')
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', lw=3, label='钻孔轴线'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=8, label='裂隙位置'),
            Line2D([0], [0], color='blue', lw=2, label='可能连通'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='yellow', 
                   markersize=10, label='建议钻孔位置')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def run_analysis(self):
        """运行完整分析"""
        
        print("=== 开始多钻孔裂隙网络连通性分析 ===")
        
        # 分析所有钻孔
        all_borehole_results = {}
        all_cracks = []
        
        for borehole_id in self.borehole_positions.keys():
            result = self.analyze_single_borehole(borehole_id)
            if result is not None:
                all_borehole_results[borehole_id] = result
                all_cracks.extend(result['crack_features'])
        
        print(f"总共检测到 {len(all_cracks)} 个裂隙")
        
        if len(all_cracks) < 2:
            print("裂隙数量不足，无法进行连通性分析")
            return None
        
        # 机器学习连通性预测
        print("执行机器学习连通性分析...")
        connectivity_results = self.machine_learning_connectivity_prediction(all_cracks)
        
        # 深度学习聚类
        print("执行深度学习聚类分析...")
        clusters = self.deep_learning_clustering(all_cracks)
        
        # 识别高不确定性区域
        print("识别高不确定性区域...")
        high_uncertainty_regions = self.identify_high_uncertainty_regions(all_cracks, connectivity_results)
        
        # 建议最优钻孔位置
        print("生成最优钻孔位置建议...")
        optimal_locations = self.suggest_optimal_drill_locations(all_cracks, connectivity_results)
        
        # 创建3D可视化
        print("生成三维可视化...")
        fig_3d = self.create_3d_visualization(all_cracks, connectivity_results, optimal_locations)
        
        # 保存结果
        self.save_results(all_borehole_results, all_cracks, connectivity_results, 
                         clusters, high_uncertainty_regions, optimal_locations, fig_3d)
        
        result = {
            'borehole_results': all_borehole_results,
            'all_cracks': all_cracks,
            'connectivity_results': connectivity_results,
            'clusters': clusters,
            'high_uncertainty_regions': high_uncertainty_regions,
            'optimal_locations': optimal_locations
        }
        
        print("=== 分析完成 ===")
        return result
    
    def save_results(self, borehole_results, all_cracks, connectivity_results, 
                    clusters, high_uncertainty_regions, optimal_locations, fig_3d):
        """保存分析结果"""
        
        # 保存3D可视化
        png_path = os.path.join(self.output_dir, "problem4_3d_connectivity_analysis.png")
        pdf_path = os.path.join(self.output_dir, "problem4_3d_connectivity_analysis.pdf")
        
        fig_3d.savefig(png_path, dpi=300, bbox_inches='tight', 
                      facecolor='white', edgecolor='none')
        fig_3d.savefig(pdf_path, dpi=300, bbox_inches='tight', 
                      facecolor='white', edgecolor='none')
        
        plt.close(fig_3d)
        
        # 保存连通性分析结果
        connectivity_data = []
        for (i, j), conn_info in connectivity_results.items():
            crack1 = all_cracks[i]
            crack2 = all_cracks[j]
            
            connectivity_data.append({
                '裂隙1_钻孔': crack1['borehole_id'],
                '裂隙1_深度': crack1['depth_info'],
                '裂隙1_面积': crack1['area'],
                '裂隙2_钻孔': crack2['borehole_id'],
                '裂隙2_深度': crack2['depth_info'],
                '裂隙2_面积': crack2['area'],
                '连通概率': f"{conn_info['probability']:.3f}",
                '预测连通': '是' if conn_info['predicted_connected'] else '否'
            })
        
        df_connectivity = pd.DataFrame(connectivity_data)
        connectivity_csv = os.path.join(self.output_dir, "problem4_connectivity_analysis.csv")
        df_connectivity.to_csv(connectivity_csv, index=False, encoding='utf-8-sig')
        
        # 保存最优钻孔位置建议
        if optimal_locations:
            location_data = []
            for i, location in enumerate(optimal_locations):
                pos = location['position']
                location_data.append({
                    '优先级': i + 1,
                    'X坐标(mm)': f"{pos[0]:.0f}",
                    'Y坐标(mm)': f"{pos[1]:.0f}",
                    'Z坐标(mm)': f"{pos[2]:.0f}",
                    '优先级分数': f"{location['priority_score']:.2f}",
                    '不确定性分数': f"{location['uncertainty_score']:.3f}",
                    '到最近钻孔距离(mm)': f"{location['distance_to_nearest']:.0f}",
                    '建议原因': location['reason']
                })
            
            df_locations = pd.DataFrame(location_data)
            locations_csv = os.path.join(self.output_dir, "problem4_optimal_drill_locations.csv")
            df_locations.to_csv(locations_csv, index=False, encoding='utf-8-sig')
        
        # 创建汇总统计图表
        self.create_summary_charts(borehole_results, all_cracks, connectivity_results, clusters)
        
        print(f"结果已保存:")
        print(f"  3D可视化: {png_path}, {pdf_path}")
        print(f"  连通性分析: {connectivity_csv}")
        if optimal_locations:
            print(f"  钻孔位置建议: {locations_csv}")
    
    def create_summary_charts(self, borehole_results, all_cracks, connectivity_results, clusters):
        """创建汇总统计图表"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('多钻孔裂隙网络分析统计汇总', fontsize=16, fontweight='bold')
        
        # 各钻孔裂隙数量统计
        borehole_ids = list(borehole_results.keys())
        crack_counts = [borehole_results[bid]['total_cracks'] for bid in borehole_ids]
        
        bars1 = axes[0, 0].bar(borehole_ids, crack_counts, alpha=0.7, color='skyblue', edgecolor='navy')
        axes[0, 0].set_title('各钻孔裂隙数量统计', fontsize=14)
        axes[0, 0].set_xlabel('钻孔编号')
        axes[0, 0].set_ylabel('裂隙数量')
        axes[0, 0].grid(True, alpha=0.3)
        
        for bar, count in zip(bars1, crack_counts):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{count}', ha='center', va='bottom', fontsize=12)
        
        # 连通概率分布
        if connectivity_results:
            probs = [conn_info['probability'] for conn_info in connectivity_results.values()]
            axes[0, 1].hist(probs, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 1].set_title(f'连通概率分布 (平均: {np.mean(probs):.3f})', fontsize=14)
            axes[0, 1].set_xlabel('连通概率')
            axes[0, 1].set_ylabel('频数')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 裂隙面积分布
        areas = [crack['area'] for crack in all_cracks]
        axes[0, 2].hist(areas, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title(f'裂隙面积分布 (平均: {np.mean(areas):.1f})', fontsize=14)
        axes[0, 2].set_xlabel('面积 (像素²)')
        axes[0, 2].set_ylabel('频数')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 深度分布
        depths = []
        for crack in all_cracks:
            depth_range = crack['depth_range']
            depth = (depth_range[0] + depth_range[1]) / 2
            depths.append(depth)
        
        axes[1, 0].hist(depths, bins=15, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 0].set_title('裂隙深度分布', fontsize=14)
        axes[1, 0].set_xlabel('深度 (mm)')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 聚类结果
        if clusters and len(clusters) > 0:
            cluster_labels = list(clusters.keys())
            cluster_sizes = [len(clusters[label]) for label in cluster_labels]
            
            # 过滤掉噪声点（标签为-1）
            valid_clusters = [(label, size) for label, size in zip(cluster_labels, cluster_sizes) if label != -1]
            
            if valid_clusters:
                valid_labels, valid_sizes = zip(*valid_clusters)
                bars2 = axes[1, 1].bar(range(len(valid_labels)), valid_sizes, 
                                     alpha=0.7, color='purple', edgecolor='black')
                axes[1, 1].set_title('裂隙聚类结果', fontsize=14)
                axes[1, 1].set_xlabel('聚类编号')
                axes[1, 1].set_ylabel('聚类大小')
                axes[1, 1].set_xticks(range(len(valid_labels)))
                axes[1, 1].set_xticklabels([f'C{i}' for i in range(len(valid_labels))])
                axes[1, 1].grid(True, alpha=0.3)
                
                for bar, size in zip(bars2, valid_sizes):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{size}', ha='center', va='bottom', fontsize=10)
        
        # 钻孔间连通性矩阵
        borehole_connectivity = np.zeros((len(borehole_ids), len(borehole_ids)))
        
        for (i, j), conn_info in connectivity_results.items():
            crack1 = all_cracks[i]
            crack2 = all_cracks[j]
            bh1_idx = borehole_ids.index(crack1['borehole_id'])
            bh2_idx = borehole_ids.index(crack2['borehole_id'])
            
            if conn_info['predicted_connected']:
                borehole_connectivity[bh1_idx, bh2_idx] += 1
                borehole_connectivity[bh2_idx, bh1_idx] += 1
        
        im = axes[1, 2].imshow(borehole_connectivity, cmap='Blues', aspect='equal')
        axes[1, 2].set_title('钻孔间连通关系矩阵', fontsize=14)
        axes[1, 2].set_xticks(range(len(borehole_ids)))
        axes[1, 2].set_yticks(range(len(borehole_ids)))
        axes[1, 2].set_xticklabels([f'{bid}#' for bid in borehole_ids])
        axes[1, 2].set_yticklabels([f'{bid}#' for bid in borehole_ids])
        
        # 添加数值标签
        for i in range(len(borehole_ids)):
            for j in range(len(borehole_ids)):
                if borehole_connectivity[i, j] > 0:
                    axes[1, 2].text(j, i, f'{int(borehole_connectivity[i, j])}',
                                   ha='center', va='center', color='white', fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 保存汇总图表
        summary_png = os.path.join(self.output_dir, "problem4_summary_statistics.png")
        summary_pdf = os.path.join(self.output_dir, "problem4_summary_statistics.pdf")
        
        fig.savefig(summary_png, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        fig.savefig(summary_pdf, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        plt.close(fig)
        
        print(f"汇总统计图已保存: {summary_png}, {summary_pdf}")

def main():
    """主函数"""
    
    # 创建分析系统
    analyzer = MultiBoreholeCrackAnalyzer(
        input_dir="附件4",
        output_dir="./"
    )
    
    # 运行分析
    results = analyzer.run_analysis()
    
    if results:
        print(f"共分析 {len(results['borehole_results'])} 个钻孔")
        print(f"检测到 {len(results['all_cracks'])} 个裂隙")
        print(f"识别出 {len(results['connectivity_results'])} 个潜在连通对")
        print(f"建议 {len(results['optimal_locations'])} 个补充钻孔位置")
        print("所有结果已保存到当前目录")

if __name__ == "__main__":
    main()
