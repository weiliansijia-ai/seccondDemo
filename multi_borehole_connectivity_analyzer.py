#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多钻孔裂隙网络连通性分析与三维重构系统
2025年中国研究生数学建模竞赛C题 - Problem4

专注于核心算法：连通性评估 + 三维重构 + 不确定性分析
"""

import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
import plotly.express as px
from scipy import optimize, signal, ndimage, interpolate
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from skimage import filters, morphology, feature, measure
import glob
import warnings
import json
warnings.filterwarnings('ignore')

# 设置专业图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['figure.titlesize'] = 16

# 专业配色
COLORS = {
    'primary': '#2E86AB',     # 主蓝色
    'secondary': '#A23B72',   # 紫红色  
    'accent': '#F18F01',      # 橙色
    'success': '#10B981',     # 翠绿
    'warning': '#F59E0B',     # 警告橙
    'danger': '#EF4444',      # 红色
    'bg_light': '#F8FAFC',    # 浅背景
    'connectivity': '#8B5CF6', # 连通性紫
}

class MultiBoreholeCrackAnalyzer:
    """多钻孔裂隙连通性分析器"""
    
    def __init__(self):
        """初始化系统"""
        self.bore_diameter = 30  # mm
        self.bore_circumference = np.pi * self.bore_diameter  # ~94.25mm
        self.depth_per_image = 1000  # mm (Problem4中每张图对应1000mm)
        self.dpi = 232.85  # Problem4的DPI
        
        # 钻孔布置参数 (2×3阵列，间距1000mm)
        self.borehole_positions = {
            '1#': {'x': 500, 'y': 2000, 'z_start': 0, 'depth': 7000},
            '2#': {'x': 1500, 'y': 2000, 'z_start': 0, 'depth': 7000},
            '3#': {'x': 2500, 'y': 2000, 'z_start': 0, 'depth': 7000},
            '4#': {'x': 500, 'y': 1000, 'z_start': 0, 'depth': 5000},
            '5#': {'x': 1500, 'y': 1000, 'z_start': 0, 'depth': 7000},
            '6#': {'x': 2500, 'y': 1000, 'z_start': 0, 'depth': 7000},
        }
        
        self.borehole_spacing = 1000  # mm
        self.all_cracks = {}  # 存储所有钻孔的裂隙数据
        
        print("多钻孔裂隙连通性分析系统初始化完成")
        print(f"钻孔直径: {self.bore_diameter}mm, 周长: {self.bore_circumference:.2f}mm")
        print(f"钻孔布置: 2×3阵列，间距: {self.borehole_spacing}mm")
    
    def load_and_preprocess_image(self, image_path):
        """加载和预处理图像"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        try:
            # 支持中文路径
            image_data = np.fromfile(image_path, dtype=np.uint8)
            original_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if original_image is None:
                raise ValueError("图像解码失败")
        except:
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为灰度
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # 预处理
        denoised = ndimage.gaussian_filter(gray_image.astype(np.float32), sigma=1.5)
        enhanced = cv2.equalizeHist(denoised.astype(np.uint8))
        
        return original_image, enhanced.astype(np.float32)
    
    def detect_cracks_in_image(self, image, borehole_id, depth_range):
        """检测单张图像中的裂隙"""
        height, width = image.shape
        detected_cracks = []
        
        # 使用多种方法检测裂隙 (复用Problem2和Problem3的方法)
        cracks_method1 = self._edge_based_detection(image)
        cracks_method2 = self._morphological_analysis(image)
        
        # 将图像坐标转换为三维空间坐标
        for crack_data in cracks_method1 + cracks_method2:
            if crack_data is not None:
                # 转换坐标
                crack_3d = self._convert_to_3d_coordinates(
                    crack_data, borehole_id, depth_range
                )
                if crack_3d:
                    detected_cracks.append(crack_3d)
        
        return detected_cracks
    
    def _edge_based_detection(self, image):
        """基于边缘检测的裂隙检测"""
        cracks = []
        
        # Canny边缘检测
        edges = cv2.Canny(image.astype(np.uint8), 30, 120)
        
        # 形态学操作
        kernel = np.ones((3, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 寻找连通组件
        labeled = measure.label(edges)
        regions = measure.regionprops(labeled)
        
        for region in regions:
            if region.area > 100:  # 过滤小区域
                coords = region.coords
                if len(coords) > 30:
                    # 按y坐标排序(深度方向)
                    coords = coords[coords[:, 0].argsort()]  
                    
                    # 计算裂隙的基本参数
                    crack_data = self._analyze_crack_geometry(coords, 'edge_detection')
                    if crack_data:
                        cracks.append(crack_data)
        
        return cracks
    
    def _morphological_analysis(self, image):
        """基于形态学分析的裂隙检测"""
        cracks = []
        
        # 开运算去除噪声
        kernel_open = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)
        
        # 顶帽运算检测明亮细节
        kernel_tophat = np.ones((7, 7), np.uint8)
        tophat = cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_TOPHAT, kernel_tophat)
        
        # 组合处理
        combined = cv2.add(opened, tophat)
        
        # 阈值分割
        _, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 150:
                # 转换轮廓格式
                coords = np.squeeze(contour)
                if len(coords.shape) == 2 and coords.shape[0] > 30:
                    # 坐标格式转换 (x,y) -> (y,x)
                    coords = coords[:, [1, 0]]  
                    crack_data = self._analyze_crack_geometry(coords, 'morphological')
                    if crack_data:
                        cracks.append(crack_data)
        
        return cracks
    
    def _analyze_crack_geometry(self, coords, method_name):
        """分析裂隙几何特征"""
        try:
            if len(coords) < 10:
                return None
            
            # 计算基本几何参数
            centroid_y = np.mean(coords[:, 0])  # 深度方向中心
            centroid_x = np.mean(coords[:, 1])  # 周向方向中心
            
            # 计算长度（沿深度方向）
            depth_span = np.max(coords[:, 0]) - np.min(coords[:, 0])
            circumfer_span = np.max(coords[:, 1]) - np.min(coords[:, 1])
            
            # 计算倾角
            if depth_span > 0:
                # 线性拟合计算倾角
                try:
                    coeffs = np.polyfit(coords[:, 0], coords[:, 1], 1)
                    inclination = np.arctan(coeffs[0]) * 180 / np.pi  # 转换为度
                except:
                    inclination = 0
            else:
                inclination = 0
            
            # 计算粗糙度 (简化版JRC)
            if len(coords) > 5:
                # 计算轮廓变化率作为粗糙度指标
                try:
                    x_data = coords[:, 0].astype(float)  # 深度
                    y_data = coords[:, 1].astype(float)  # 周向
                    
                    # 去重并排序
                    unique_indices = np.unique(x_data, return_index=True)[1]
                    x_unique = x_data[unique_indices]
                    y_unique = y_data[unique_indices]
                    
                    if len(x_unique) > 5:
                        sort_indices = np.argsort(x_unique)
                        x_sorted = x_unique[sort_indices]
                        y_sorted = y_unique[sort_indices]
                        
                        # 计算Z2参数
                        slopes = []
                        for i in range(len(x_sorted) - 1):
                            dx = x_sorted[i+1] - x_sorted[i]
                            dy = y_sorted[i+1] - y_sorted[i]
                            if abs(dx) > 1e-10:
                                slopes.append((dy / dx) ** 2)
                        
                        if slopes:
                            z2 = np.sqrt(np.mean(slopes))
                            jrc = max(0, 51.85 * (z2 ** 0.6) - 10.37)
                        else:
                            jrc = 0
                    else:
                        jrc = 0
                except:
                    jrc = 0
            else:
                jrc = 0
            
            return {
                'coords': coords,
                'centroid_depth': centroid_y,
                'centroid_circumference': centroid_x,
                'depth_span': depth_span,
                'circumference_span': circumfer_span,
                'inclination': inclination,
                'jrc': jrc,
                'method': method_name,
                'crack_length': np.sqrt(depth_span**2 + circumfer_span**2)
            }
            
        except Exception as e:
            print(f"几何分析失败 ({method_name}): {e}")
            return None
    
    def _convert_to_3d_coordinates(self, crack_data, borehole_id, depth_range):
        """将图像坐标转换为三维空间坐标"""
        try:
            borehole_info = self.borehole_positions[borehole_id]
            
            # 图像坐标到物理坐标的转换
            # 深度方向: 图像像素 -> 实际深度(mm)
            depth_start, depth_end = depth_range
            image_height = crack_data['coords'][:, 0].max() - crack_data['coords'][:, 0].min() + 1
            
            # 转换深度坐标
            actual_depth = depth_start + (crack_data['centroid_depth'] / image_height) * (depth_end - depth_start)
            z_coordinate = borehole_info['z_start'] - actual_depth  # z轴向上为正，深度向下
            
            # 周向坐标转换
            circumference_ratio = crack_data['centroid_circumference'] / self.bore_circumference * 100
            # 假设裂隙在钻孔壁面，计算x,y坐标
            angle = (circumference_ratio / 100) * 2 * np.pi  # 转换为弧度
            radius = self.bore_diameter / 2
            
            # 相对于钻孔中心的坐标
            x_rel = radius * np.cos(angle)
            y_rel = radius * np.sin(angle)
            
            # 绝对坐标
            x_absolute = borehole_info['x'] + x_rel
            y_absolute = borehole_info['y'] + y_rel
            
            # 创建三维裂隙数据
            crack_3d = crack_data.copy()
            crack_3d.update({
                'borehole_id': borehole_id,
                'x': x_absolute,
                'y': y_absolute,
                'z': z_coordinate,
                'depth_range': depth_range,
                'angle': angle,
                'spatial_extent': {
                    'x_min': x_absolute - crack_data['circumference_span'] / 2,
                    'x_max': x_absolute + crack_data['circumference_span'] / 2,
                    'y_min': y_absolute - crack_data['circumference_span'] / 2,
                    'y_max': y_absolute + crack_data['circumference_span'] / 2,
                    'z_min': z_coordinate - crack_data['depth_span'] / 2,
                    'z_max': z_coordinate + crack_data['depth_span'] / 2
                }
            })
            
            return crack_3d
            
        except Exception as e:
            print(f"三维坐标转换失败: {e}")
            return None
    
    def process_all_boreholes(self):
        """处理所有钻孔数据"""
        print("\n🔍 开始处理所有钻孔数据...")
        
        # 获取附件4数据路径
        data_folder = "附件4"
        
        if not os.path.exists(data_folder):
            print(f"❌ 数据文件夹不存在: {data_folder}")
            return
        
        total_cracks = 0
        
        for borehole_id in self.borehole_positions.keys():
            print(f"\n📊 处理钻孔 {borehole_id}...")
            borehole_folder = os.path.join(data_folder, f"{borehole_id}孔")
            
            if not os.path.exists(borehole_folder):
                print(f"❌ 钻孔文件夹不存在: {borehole_folder}")
                continue
            
            # 获取该钻孔的所有图像文件
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(glob.glob(os.path.join(borehole_folder, ext)))
            
            image_files.sort()  # 按文件名排序
            
            borehole_cracks = []
            
            for i, image_file in enumerate(image_files):
                # 计算深度范围
                depth_start = i * self.depth_per_image
                depth_end = (i + 1) * self.depth_per_image
                
                print(f"  处理图像 {os.path.basename(image_file)} (深度 {depth_start}-{depth_end}mm)")
                
                try:
                    # 加载图像
                    original_image, processed_image = self.load_and_preprocess_image(image_file)
                    
                    # 检测裂隙
                    image_cracks = self.detect_cracks_in_image(
                        processed_image, borehole_id, (depth_start, depth_end)
                    )
                    
                    borehole_cracks.extend(image_cracks)
                    
                except Exception as e:
                    print(f"    ❌ 处理失败: {e}")
                    continue
            
            self.all_cracks[borehole_id] = borehole_cracks
            total_cracks += len(borehole_cracks)
            print(f"  ✅ 钻孔 {borehole_id} 检测到 {len(borehole_cracks)} 条裂隙")
        
        print(f"\n🎉 所有钻孔处理完成！")
        print(f"总共检测到 {total_cracks} 条裂隙")
        
        return self.all_cracks
    
    def calculate_connectivity_probability(self, crack1, crack2):
        """计算两条裂隙的连通概率"""
        try:
            # 距离因子
            distance = np.sqrt(
                (crack1['x'] - crack2['x'])**2 + 
                (crack1['y'] - crack2['y'])**2 + 
                (crack1['z'] - crack2['z'])**2
            )
            
            # 基础距离衰减 (距离越远，连通概率越低)
            max_connection_distance = 2000  # mm
            distance_factor = max(0, 1 - distance / max_connection_distance)
            
            # 方向因子 (倾角越相似，连通概率越高)
            angle_diff = abs(crack1['inclination'] - crack2['inclination'])
            angle_factor = max(0, 1 - angle_diff / 90)  # 90度为最大差异
            
            # 尺寸因子 (裂隙越大，连通概率越高)
            size1 = crack1['crack_length']
            size2 = crack2['crack_length']
            avg_size = (size1 + size2) / 2
            size_factor = min(1, avg_size / 100)  # 100mm为基准尺寸
            
            # 粗糙度因子 (JRC越高，连通阻力越大)
            jrc1 = crack1.get('jrc', 0)
            jrc2 = crack2.get('jrc', 0)
            avg_jrc = (jrc1 + jrc2) / 2
            roughness_factor = max(0.1, 1 - avg_jrc / 100)  # JRC=100时因子为0.1
            
            # 深度对齐因子
            z_diff = abs(crack1['z'] - crack2['z'])
            depth_factor = max(0, 1 - z_diff / 500)  # 500mm深度差异为基准
            
            # 综合连通概率
            connectivity_prob = (
                distance_factor * 0.4 +      # 距离权重40%
                angle_factor * 0.2 +         # 方向权重20%
                size_factor * 0.15 +         # 尺寸权重15%
                roughness_factor * 0.15 +    # 粗糙度权重15%
                depth_factor * 0.1           # 深度权重10%
            )
            
            return max(0, min(1, connectivity_prob))
            
        except Exception as e:
            print(f"连通概率计算失败: {e}")
            return 0
    
    def analyze_connectivity(self):
        """分析裂隙连通性"""
        print("\n🔗 开始连通性分析...")
        
        if not self.all_cracks:
            print("❌ 无裂隙数据，请先运行处理所有钻孔数据")
            return None
        
        connectivity_results = []
        connectivity_matrix = {}
        
        # 获取所有裂隙对
        all_crack_pairs = []
        borehole_ids = list(self.all_cracks.keys())
        
        for i, borehole_id1 in enumerate(borehole_ids):
            for j, borehole_id2 in enumerate(borehole_ids):
                if i <= j:  # 避免重复比较
                    continue
                    
                cracks1 = self.all_cracks[borehole_id1]
                cracks2 = self.all_cracks[borehole_id2]
                
                for crack1 in cracks1:
                    for crack2 in cracks2:
                        prob = self.calculate_connectivity_probability(crack1, crack2)
                        if prob > 0.1:  # 只记录连通概率大于0.1的裂隙对
                            connection = {
                                'borehole1': borehole_id1,
                                'borehole2': borehole_id2,
                                'crack1': crack1,
                                'crack2': crack2,
                                'probability': prob,
                                'distance': np.sqrt(
                                    (crack1['x'] - crack2['x'])**2 + 
                                    (crack1['y'] - crack2['y'])**2 + 
                                    (crack1['z'] - crack2['z'])**2
                                )
                            }
                            connectivity_results.append(connection)
        
        # 排序连通性结果
        connectivity_results.sort(key=lambda x: x['probability'], reverse=True)
        
        print(f"✅ 发现 {len(connectivity_results)} 对可能连通的裂隙")
        
        # 统计高概率连通
        high_prob_connections = [c for c in connectivity_results if c['probability'] > 0.7]
        medium_prob_connections = [c for c in connectivity_results if 0.4 <= c['probability'] <= 0.7]
        low_prob_connections = [c for c in connectivity_results if 0.1 <= c['probability'] < 0.4]
        
        print(f"  🔥 高概率连通 (>0.7): {len(high_prob_connections)} 对")
        print(f"  🟡 中等概率连通 (0.4-0.7): {len(medium_prob_connections)} 对")
        print(f"  🔵 低概率连通 (0.1-0.4): {len(low_prob_connections)} 对")
        
        return {
            'all_connections': connectivity_results,
            'high_prob': high_prob_connections,
            'medium_prob': medium_prob_connections,
            'low_prob': low_prob_connections,
            'statistics': {
                'total_connections': len(connectivity_results),
                'high_prob_count': len(high_prob_connections),
                'medium_prob_count': len(medium_prob_connections),
                'low_prob_count': len(low_prob_connections)
            }
        }
    
    def identify_uncertainty_regions(self, connectivity_results):
        """识别高不确定性区域"""
        print("\n🎯 识别高不确定性区域...")
        
        # 定义空间网格 (1m × 1m × 0.5m)
        x_range = np.arange(0, 3000, 500)  # 500mm网格
        y_range = np.arange(500, 2500, 500)
        z_range = np.arange(-7000, 500, 500)
        
        uncertainty_grid = np.zeros((len(x_range)-1, len(y_range)-1, len(z_range)-1))
        coverage_grid = np.zeros((len(x_range)-1, len(y_range)-1, len(z_range)-1))
        
        # 计算每个网格单元的不确定性
        for i in range(len(x_range)-1):
            for j in range(len(y_range)-1):
                for k in range(len(z_range)-1):
                    x_center = (x_range[i] + x_range[i+1]) / 2
                    y_center = (y_range[j] + y_range[j+1]) / 2
                    z_center = (z_range[k] + z_range[k+1]) / 2
                    
                    # 计算该区域的观测密度
                    nearby_cracks = []
                    for borehole_id, cracks in self.all_cracks.items():
                        for crack in cracks:
                            distance = np.sqrt(
                                (crack['x'] - x_center)**2 + 
                                (crack['y'] - y_center)**2 + 
                                (crack['z'] - z_center)**2
                            )
                            if distance < 750:  # 750mm影响半径
                                nearby_cracks.append(crack)
                    
                    # 计算连通性信息密度
                    nearby_connections = []
                    for conn in connectivity_results['all_connections']:
                        avg_x = (conn['crack1']['x'] + conn['crack2']['x']) / 2
                        avg_y = (conn['crack1']['y'] + conn['crack2']['y']) / 2
                        avg_z = (conn['crack1']['z'] + conn['crack2']['z']) / 2
                        
                        distance = np.sqrt(
                            (avg_x - x_center)**2 + 
                            (avg_y - y_center)**2 + 
                            (avg_z - z_center)**2
                        )
                        if distance < 750:
                            nearby_connections.append(conn)
                    
                    # 不确定性评估
                    crack_density = len(nearby_cracks)
                    connection_density = len(nearby_connections)
                    
                    # 观测覆盖度 (基于钻孔距离)
                    min_borehole_distance = float('inf')
                    for borehole_id, info in self.borehole_positions.items():
                        distance = np.sqrt(
                            (info['x'] - x_center)**2 + 
                            (info['y'] - y_center)**2
                        )
                        min_borehole_distance = min(min_borehole_distance, distance)
                    
                    # 归一化覆盖度 (距离越远覆盖度越低)
                    coverage = max(0, 1 - min_borehole_distance / 1500)
                    coverage_grid[i, j, k] = coverage
                    
                    # 不确定性计算 (覆盖度低 + 连通信息少 = 不确定性高)
                    if coverage > 0:
                        uncertainty = (1 - coverage) * 0.6 + (1 - min(1, connection_density/5)) * 0.4
                    else:
                        uncertainty = 1.0  # 完全无观测区域
                    
                    uncertainty_grid[i, j, k] = uncertainty
        
        # 找到高不确定性区域
        high_uncertainty_threshold = 0.6
        high_uncertainty_indices = np.where(uncertainty_grid > high_uncertainty_threshold)
        
        uncertainty_regions = []
        for idx in range(len(high_uncertainty_indices[0])):
            i, j, k = high_uncertainty_indices[0][idx], high_uncertainty_indices[1][idx], high_uncertainty_indices[2][idx]
            
            region = {
                'x': (x_range[i] + x_range[i+1]) / 2,
                'y': (y_range[j] + y_range[j+1]) / 2,
                'z': (z_range[k] + z_range[k+1]) / 2,
                'uncertainty_score': uncertainty_grid[i, j, k],
                'coverage_score': coverage_grid[i, j, k],
                'grid_index': (i, j, k)
            }
            uncertainty_regions.append(region)
        
        # 按不确定性得分排序
        uncertainty_regions.sort(key=lambda x: x['uncertainty_score'], reverse=True)
        
        print(f"✅ 识别出 {len(uncertainty_regions)} 个高不确定性区域")
        
        return {
            'uncertainty_grid': uncertainty_grid,
            'coverage_grid': coverage_grid,
            'high_uncertainty_regions': uncertainty_regions[:20],  # 前20个最高不确定性区域
            'grid_params': {
                'x_range': x_range,
                'y_range': y_range, 
                'z_range': z_range
            }
        }
    
    def optimize_additional_boreholes(self, uncertainty_analysis):
        """优化补充钻孔位置"""
        print("\n⚡ 优化补充钻孔位置...")
        
        # 候选位置评估
        candidate_positions = []
        
        # 基于高不确定性区域生成候选位置
        for region in uncertainty_analysis['high_uncertainty_regions']:
            # 只考虑地表附近的位置 (z > -1000mm)
            if region['z'] > -1000:
                candidate = {
                    'x': region['x'],
                    'y': region['y'],
                    'uncertainty_reduction': region['uncertainty_score'],
                    'coverage_improvement': 1 - region['coverage_score'],
                    'feasibility': self._assess_drilling_feasibility(region['x'], region['y'])
                }
                candidate_positions.append(candidate)
        
        # 添加系统性优化位置
        # 在现有钻孔的中间位置
        systematic_candidates = [
            {'x': 1000, 'y': 1500, 'description': '钻孔1-2与4-5中心'},
            {'x': 2000, 'y': 1500, 'description': '钻孔2-3与5-6中心'},
            {'x': 1500, 'y': 1500, 'description': '整体布局中心'},
            {'x': 750, 'y': 1500, 'description': '1-4钻孔延长线'},
            {'x': 2250, 'y': 1500, 'description': '3-6钻孔延长线'}
        ]
        
        for candidate in systematic_candidates:
            candidate.update({
                'uncertainty_reduction': self._estimate_uncertainty_reduction(candidate, uncertainty_analysis),
                'coverage_improvement': self._estimate_coverage_improvement(candidate),
                'feasibility': self._assess_drilling_feasibility(candidate['x'], candidate['y'])
            })
            candidate_positions.append(candidate)
        
        # 综合评分
        for candidate in candidate_positions:
            candidate['total_score'] = (
                candidate['uncertainty_reduction'] * 0.4 +
                candidate['coverage_improvement'] * 0.3 +
                candidate['feasibility'] * 0.3
            )
        
        # 排序并选择前3个
        candidate_positions.sort(key=lambda x: x['total_score'], reverse=True)
        optimal_positions = candidate_positions[:3]
        
        print("✅ 优化完成，推荐补充钻孔位置:")
        for i, pos in enumerate(optimal_positions, 1):
            print(f"  {i}. 位置 ({pos['x']:.0f}, {pos['y']:.0f}) - 综合评分: {pos['total_score']:.3f}")
            if 'description' in pos:
                print(f"     说明: {pos['description']}")
        
        return optimal_positions
    
    def _assess_drilling_feasibility(self, x, y):
        """评估钻孔施工可行性"""
        # 基于与现有钻孔的距离、边界条件等
        
        # 检查边界约束 (留出500mm安全距离)
        if x < 500 or x > 2500 or y < 500 or y > 2500:
            return 0.2  # 边界附近可行性低
        
        # 检查与现有钻孔的距离
        min_distance = float('inf')
        for borehole_id, info in self.borehole_positions.items():
            distance = np.sqrt((info['x'] - x)**2 + (info['y'] - y)**2)
            min_distance = min(min_distance, distance)
        
        # 最优距离为500-800mm
        if 500 <= min_distance <= 800:
            return 1.0
        elif min_distance < 500:
            return 0.5  # 太近
        else:
            return max(0.3, 1 - (min_distance - 800) / 1000)  # 距离惩罚
    
    def _estimate_uncertainty_reduction(self, candidate, uncertainty_analysis):
        """估计新钻孔对不确定性的降低效果"""
        reduction_score = 0
        for region in uncertainty_analysis['high_uncertainty_regions'][:10]:
            distance = np.sqrt(
                (region['x'] - candidate['x'])**2 + 
                (region['y'] - candidate['y'])**2
            )
            if distance < 750:  # 影响范围
                weight = max(0, 1 - distance / 750)
                reduction_score += region['uncertainty_score'] * weight
        
        return min(1, reduction_score / 5)
    
    def _estimate_coverage_improvement(self, candidate):
        """估计新钻孔对覆盖度的改善"""
        # 计算新钻孔能覆盖的未覆盖区域
        coverage_radius = 750  # mm
        
        # 简化计算：基于与现有钻孔形成的几何覆盖
        existing_positions = [(info['x'], info['y']) for info in self.borehole_positions.values()]
        new_coverage_area = 0
        
        # 估算新增覆盖面积
        for i in range(-1, 2):  # 3x3网格采样
            for j in range(-1, 2):
                test_x = candidate['x'] + i * 250
                test_y = candidate['y'] + j * 250
                
                # 检查是否在现有覆盖范围内
                covered = False
                for ex_x, ex_y in existing_positions:
                    if np.sqrt((test_x - ex_x)**2 + (test_y - ex_y)**2) < coverage_radius:
                        covered = True
                        break
                
                if not covered:
                    new_coverage_area += 1
        
        return min(1, new_coverage_area / 9)  # 归一化
    
    def create_3d_visualization(self, connectivity_results, uncertainty_analysis, save_path="Problem4/results/"):
        """创建三维可视化"""
        print("\n🎨 创建三维可视化...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 使用Plotly创建交互式3D可视化
        fig = go.Figure()
        
        # 1. 添加钻孔位置
        borehole_x = [info['x'] for info in self.borehole_positions.values()]
        borehole_y = [info['y'] for info in self.borehole_positions.values()]
        borehole_z = [info['z_start'] for info in self.borehole_positions.values()]
        borehole_names = list(self.borehole_positions.keys())
        
        fig.add_trace(go.Scatter3d(
            x=borehole_x, y=borehole_y, z=borehole_z,
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='diamond'),
            text=borehole_names,
            textposition='top center',
            name='钻孔位置',
            hovertemplate='<b>%{text}</b><br>坐标: (%{x}, %{y}, %{z})<extra></extra>'
        ))
        
        # 2. 添加裂隙点
        for borehole_id, cracks in self.all_cracks.items():
            if cracks:
                crack_x = [crack['x'] for crack in cracks]
                crack_y = [crack['y'] for crack in cracks]
                crack_z = [crack['z'] for crack in cracks]
                crack_jrc = [crack.get('jrc', 0) for crack in cracks]
                
                fig.add_trace(go.Scatter3d(
                    x=crack_x, y=crack_y, z=crack_z,
                    mode='markers',
                    marker=dict(
                        size=6, 
                        color=crack_jrc, 
                        colorscale='Viridis',
                        colorbar=dict(title='JRC粗糙度'),
                        showscale=True if borehole_id == '1#' else False
                    ),
                    name=f'裂隙-{borehole_id}',
                    hovertemplate=f'<b>钻孔 {borehole_id}</b><br>位置: (%{{x:.0f}}, %{{y:.0f}}, %{{z:.0f}})<br>JRC: %{{marker.color:.1f}}<extra></extra>'
                ))
        
        # 3. 添加高概率连通线
        for i, conn in enumerate(connectivity_results['high_prob'][:20]):  # 只显示前20个
            fig.add_trace(go.Scatter3d(
                x=[conn['crack1']['x'], conn['crack2']['x']],
                y=[conn['crack1']['y'], conn['crack2']['y']],
                z=[conn['crack1']['z'], conn['crack2']['z']],
                mode='lines',
                line=dict(
                    color='rgba(255, 0, 0, 0.8)',
                    width=4
                ),
                name='高概率连通' if i == 0 else None,
                showlegend=True if i == 0 else False,
                hovertemplate=f'<b>连通概率: {conn["probability"]:.3f}</b><br>距离: {conn["distance"]:.1f}mm<extra></extra>'
            ))
        
        # 4. 添加中等概率连通线
        for i, conn in enumerate(connectivity_results['medium_prob'][:10]):
            fig.add_trace(go.Scatter3d(
                x=[conn['crack1']['x'], conn['crack2']['x']],
                y=[conn['crack1']['y'], conn['crack2']['y']],
                z=[conn['crack1']['z'], conn['crack2']['z']],
                mode='lines',
                line=dict(
                    color='rgba(255, 165, 0, 0.6)',
                    width=2
                ),
                name='中等概率连通' if i == 0 else None,
                showlegend=True if i == 0 else False,
                hovertemplate=f'<b>连通概率: {conn["probability"]:.3f}</b><br>距离: {conn["distance"]:.1f}mm<extra></extra>'
            ))
        
        # 5. 添加高不确定性区域
        uncertainty_regions = uncertainty_analysis['high_uncertainty_regions'][:15]
        if uncertainty_regions:
            uncertain_x = [region['x'] for region in uncertainty_regions]
            uncertain_y = [region['y'] for region in uncertainty_regions]
            uncertain_z = [region['z'] for region in uncertainty_regions]
            uncertain_scores = [region['uncertainty_score'] for region in uncertainty_regions]
            
            fig.add_trace(go.Scatter3d(
                x=uncertain_x, y=uncertain_y, z=uncertain_z,
                mode='markers',
                marker=dict(
                    size=10,
                    color=uncertain_scores,
                    colorscale='Reds',
                    opacity=0.6,
                    colorbar=dict(title='不确定性得分', x=1.1)
                ),
                name='高不确定性区域',
                hovertemplate='<b>不确定性区域</b><br>位置: (%{x:.0f}, %{y:.0f}, %{z:.0f})<br>不确定性: %{marker.color:.3f}<extra></extra>'
            ))
        
        # 设置布局
        fig.update_layout(
            title={
                'text': '多钻孔裂隙网络连通性三维分析<br>Multi-Borehole Crack Network Connectivity Analysis',
                'x': 0.5,
                'font': dict(size=16)
            },
            scene=dict(
                xaxis_title='X坐标 (mm)',
                yaxis_title='Y坐标 (mm)',
                zaxis_title='Z坐标 (mm)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1200,
            height=800,
            font=dict(family='Arial', size=12),
            legend=dict(x=0.02, y=0.98)
        )
        
        # 保存交互式HTML
        html_path = os.path.join(save_path, "3d_connectivity_analysis.html")
        fig.write_html(html_path)
        
        # 保存静态图像
        png_path = os.path.join(save_path, "3d_connectivity_analysis.png")
        try:
            fig.write_image(png_path, width=1200, height=800, scale=2)
        except:
            print("⚠️  静态图像保存需要安装kaleido: pip install kaleido")
        
        print(f"✅ 三维可视化已保存:")
        print(f"   交互式版本: {html_path}")
        if os.path.exists(png_path):
            print(f"   静态图像: {png_path}")
        
        return fig


def main():
    """主函数"""
    print("="*80)
    print("🎯 多钻孔裂隙网络连通性分析与三维重构系统")
    print("2025年中国研究生数学建模竞赛C题 - Problem4")
    print("专注于核心算法：连通性评估 + 三维重构 + 不确定性分析")
    print("="*80)
    
    # 初始化系统
    analyzer = MultiBoreholeCrackAnalyzer()
    
    # 1. 处理所有钻孔数据
    print("\n" + "="*50)
    print("第一步：处理所有钻孔数据")
    print("="*50)
    all_cracks = analyzer.process_all_boreholes()
    
    if not all_cracks:
        print("❌ 无法获取裂隙数据，程序退出")
        return
    
    # 2. 连通性分析
    print("\n" + "="*50)
    print("第二步：连通性分析")
    print("="*50)
    connectivity_results = analyzer.analyze_connectivity()
    
    if not connectivity_results:
        print("❌ 连通性分析失败，程序退出")
        return
    
    # 3. 不确定性分析
    print("\n" + "="*50)
    print("第三步：不确定性分析")
    print("="*50)
    uncertainty_analysis = analyzer.identify_uncertainty_regions(connectivity_results)
    
    # 4. 钻孔布局优化
    print("\n" + "="*50)
    print("第四步：补充钻孔优化")
    print("="*50)
    optimal_positions = analyzer.optimize_additional_boreholes(uncertainty_analysis)
    
    # 5. 三维可视化
    print("\n" + "="*50)
    print("第五步：三维可视化")
    print("="*50)
    visualization = analyzer.create_3d_visualization(connectivity_results, uncertainty_analysis)
    
    # 6. 生成结果报告
    print("\n" + "="*50)
    print("分析结果汇总")
    print("="*50)
    
    print(f"📊 检测结果统计:")
    total_cracks = sum(len(cracks) for cracks in all_cracks.values())
    print(f"  • 总裂隙数: {total_cracks} 条")
    print(f"  • 钻孔数: {len(all_cracks)} 个")
    
    print(f"\n🔗 连通性分析结果:")
    stats = connectivity_results['statistics']
    print(f"  • 总连通关系: {stats['total_connections']} 对")
    print(f"  • 高概率连通: {stats['high_prob_count']} 对")
    print(f"  • 中等概率连通: {stats['medium_prob_count']} 对")
    print(f"  • 低概率连通: {stats['low_prob_count']} 对")
    
    print(f"\n🎯 不确定性分析结果:")
    print(f"  • 高不确定性区域: {len(uncertainty_analysis['high_uncertainty_regions'])} 个")
    print(f"  • 最高不确定性得分: {max(r['uncertainty_score'] for r in uncertainty_analysis['high_uncertainty_regions']):.3f}")
    
    print(f"\n⚡ 优化建议:")
    for i, pos in enumerate(optimal_positions, 1):
        print(f"  {i}. 补充钻孔位置: ({pos['x']:.0f}, {pos['y']:.0f}) - 评分: {pos['total_score']:.3f}")
    
    print(f"\n📁 输出文件:")
    print("  1. Problem4/results/3d_connectivity_analysis.html - 交互式三维可视化")
    print("  2. Problem4/results/3d_connectivity_analysis.png - 静态三维图像")
    
    print("\n" + "="*80)
    print("🎉 Problem4 多钻孔裂隙网络连通性分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()
