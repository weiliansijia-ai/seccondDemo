#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem4 快速可视化程序
快速生成多钻孔裂隙网络的三维可视化
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import glob
from scipy import ndimage
from skimage import measure
import warnings
warnings.filterwarnings('ignore')

class QuickCrackVisualizer:
    """快速裂隙可视化器"""
    
    def __init__(self):
        """初始化"""
        self.bore_diameter = 30  # mm
        self.bore_circumference = np.pi * self.bore_diameter
        self.depth_per_image = 1000  # mm
        
        # 钻孔位置
        self.borehole_positions = {
            '1#': {'x': 500, 'y': 2000, 'z_start': 0, 'depth': 7000},
            '2#': {'x': 1500, 'y': 2000, 'z_start': 0, 'depth': 7000},
            '3#': {'x': 2500, 'y': 2000, 'z_start': 0, 'depth': 7000},
            '4#': {'x': 500, 'y': 1000, 'z_start': 0, 'depth': 5000},
            '5#': {'x': 1500, 'y': 1000, 'z_start': 0, 'depth': 7000},
            '6#': {'x': 2500, 'y': 1000, 'z_start': 0, 'depth': 7000},
        }
        
        self.cracks_3d = {}
        
    def quick_crack_detection(self, image):
        """快速裂隙检测"""
        # 简化的边缘检测
        edges = cv2.Canny(image.astype(np.uint8), 50, 150)
        
        # 形态学处理
        kernel = np.ones((3, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 连通组件分析
        labeled = measure.label(edges)
        regions = measure.regionprops(labeled)
        
        cracks = []
        for region in regions:
            if region.area > 50:  # 面积过滤
                coords = region.coords
                if len(coords) > 20:
                    # 计算中心点
                    centroid_y = np.mean(coords[:, 0])
                    centroid_x = np.mean(coords[:, 1])
                    
                    crack_data = {
                        'centroid_depth': centroid_y,
                        'centroid_circumference': centroid_x,
                        'area': region.area,
                        'crack_length': np.sqrt(region.major_axis_length**2 + region.minor_axis_length**2)
                    }
                    cracks.append(crack_data)
        
        return cracks
    
    def convert_to_3d(self, crack_data, borehole_id, depth_range):
        """转换为三维坐标"""
        try:
            borehole_info = self.borehole_positions[borehole_id]
            
            # 深度转换
            depth_start, depth_end = depth_range
            actual_depth = depth_start + (crack_data['centroid_depth'] / 1000) * (depth_end - depth_start)
            z_coordinate = borehole_info['z_start'] - actual_depth
            
            # 周向转换
            angle = (crack_data['centroid_circumference'] / 100) * 2 * np.pi
            radius = self.bore_diameter / 2
            
            x_rel = radius * np.cos(angle)
            y_rel = radius * np.sin(angle)
            
            x_absolute = borehole_info['x'] + x_rel
            y_absolute = borehole_info['y'] + y_rel
            
            return {
                'x': x_absolute,
                'y': y_absolute,
                'z': z_coordinate,
                'borehole_id': borehole_id,
                'area': crack_data['area'],
                'length': crack_data['crack_length']
            }
        except:
            return None
    
    def process_boreholes_quick(self, max_images_per_borehole=3):
        """快速处理钻孔数据"""
        print("🚀 开始快速处理钻孔数据...")
        
        data_folder = "附件4"
        if not os.path.exists(data_folder):
            print(f"❌ 数据文件夹不存在: {data_folder}")
            return
        
        total_cracks = 0
        
        for borehole_id in self.borehole_positions.keys():
            print(f"📊 处理钻孔 {borehole_id}...")
            borehole_folder = os.path.join(data_folder, f"{borehole_id}孔")
            
            if not os.path.exists(borehole_folder):
                continue
            
            # 获取图像文件
            image_files = glob.glob(os.path.join(borehole_folder, "*.jpg"))
            image_files.sort()
            
            # 只处理前几张图像（加速）
            image_files = image_files[:max_images_per_borehole]
            
            borehole_cracks = []
            
            for i, image_file in enumerate(image_files):
                depth_start = i * self.depth_per_image
                depth_end = (i + 1) * self.depth_per_image
                
                try:
                    # 读取图像
                    image_data = np.fromfile(image_file, dtype=np.uint8)
                    image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                    
                    # 检测裂隙
                    cracks = self.quick_crack_detection(image)
                    
                    # 随机采样（减少数据量）
                    if len(cracks) > 100:
                        indices = np.random.choice(len(cracks), 100, replace=False)
                        cracks = [cracks[i] for i in indices]
                    
                    # 转换为3D坐标
                    for crack_data in cracks:
                        crack_3d = self.convert_to_3d(crack_data, borehole_id, (depth_start, depth_end))
                        if crack_3d:
                            borehole_cracks.append(crack_3d)
                    
                except Exception as e:
                    print(f"    ❌ 处理失败: {e}")
                    continue
            
            self.cracks_3d[borehole_id] = borehole_cracks
            total_cracks += len(borehole_cracks)
            print(f"  ✅ 钻孔 {borehole_id} 检测到 {len(borehole_cracks)} 条裂隙")
        
        print(f"🎉 快速处理完成！总共检测到 {total_cracks} 条裂隙")
        return self.cracks_3d
    
    def calculate_simple_connectivity(self):
        """简化的连通性计算"""
        print("🔗 计算裂隙连通性...")
        
        connections = []
        borehole_ids = list(self.cracks_3d.keys())
        
        for i, borehole_id1 in enumerate(borehole_ids):
            for j, borehole_id2 in enumerate(borehole_ids):
                if i >= j:
                    continue
                
                cracks1 = self.cracks_3d[borehole_id1]
                cracks2 = self.cracks_3d[borehole_id2]
                
                # 只计算部分连通关系（采样）
                sample_size = min(50, len(cracks1), len(cracks2))
                if sample_size < 10:
                    continue
                
                indices1 = np.random.choice(len(cracks1), sample_size, replace=False)
                indices2 = np.random.choice(len(cracks2), sample_size, replace=False)
                
                for idx1 in indices1:
                    for idx2 in indices2:
                        crack1 = cracks1[idx1]
                        crack2 = cracks2[idx2]
                        
                        # 计算距离
                        distance = np.sqrt(
                            (crack1['x'] - crack2['x'])**2 + 
                            (crack1['y'] - crack2['y'])**2 + 
                            (crack1['z'] - crack2['z'])**2
                        )
                        
                        # 简化的连通概率
                        if distance < 1500:  # 1.5m内认为可能连通
                            prob = max(0, 1 - distance / 1500)
                            if prob > 0.3:
                                connections.append({
                                    'borehole1': borehole_id1,
                                    'borehole2': borehole_id2,
                                    'crack1': crack1,
                                    'crack2': crack2,
                                    'probability': prob,
                                    'distance': distance
                                })
        
        # 排序并返回前100个
        connections.sort(key=lambda x: x['probability'], reverse=True)
        print(f"✅ 发现 {len(connections)} 对可能连通的裂隙")
        
        return connections[:100]  # 只返回前100个
    
    def create_3d_plot(self, connections):
        """创建三维图表"""
        print("🎨 创建三维可视化...")
        
        # 创建Plotly图表
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
            name='钻孔位置'
        ))
        
        # 2. 添加裂隙点
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        for i, (borehole_id, cracks) in enumerate(self.cracks_3d.items()):
            if cracks:
                crack_x = [crack['x'] for crack in cracks]
                crack_y = [crack['y'] for crack in cracks]
                crack_z = [crack['z'] for crack in cracks]
                
                fig.add_trace(go.Scatter3d(
                    x=crack_x, y=crack_y, z=crack_z,
                    mode='markers',
                    marker=dict(size=4, color=colors[i % len(colors)], opacity=0.6),
                    name=f'裂隙-{borehole_id}'
                ))
        
        # 3. 添加连通线
        for i, conn in enumerate(connections[:20]):  # 只显示前20个
            fig.add_trace(go.Scatter3d(
                x=[conn['crack1']['x'], conn['crack2']['x']],
                y=[conn['crack1']['y'], conn['crack2']['y']],
                z=[conn['crack1']['z'], conn['crack2']['z']],
                mode='lines',
                line=dict(color='red', width=3),
                name='连通关系' if i == 0 else None,
                showlegend=True if i == 0 else False
            ))
        
        # 设置布局
        fig.update_layout(
            title='多钻孔裂隙网络三维可视化',
            scene=dict(
                xaxis_title='X坐标 (mm)',
                yaxis_title='Y坐标 (mm)',
                zaxis_title='Z坐标 (mm)',
                aspectmode='data'
            ),
            width=1200,
            height=800
        )
        
        return fig
    
    def create_summary_plot(self):
        """创建汇总统计图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Problem4 多钻孔裂隙分析汇总', fontsize=16, fontweight='bold')
        
        # 1. 各钻孔裂隙数量
        borehole_names = list(self.cracks_3d.keys())
        crack_counts = [len(cracks) for cracks in self.cracks_3d.values()]
        
        axes[0, 0].bar(borehole_names, crack_counts, color='steelblue', alpha=0.7)
        axes[0, 0].set_title('各钻孔检测到的裂隙数量')
        axes[0, 0].set_ylabel('裂隙数量')
        
        # 2. 裂隙空间分布
        all_x = []
        all_y = []
        all_z = []
        for cracks in self.cracks_3d.values():
            all_x.extend([crack['x'] for crack in cracks])
            all_y.extend([crack['y'] for crack in cracks])
            all_z.extend([crack['z'] for crack in cracks])
        
        scatter = axes[0, 1].scatter(all_x, all_y, c=all_z, cmap='viridis', alpha=0.6, s=10)
        axes[0, 1].set_title('裂隙平面分布 (颜色=深度)')
        axes[0, 1].set_xlabel('X坐标 (mm)')
        axes[0, 1].set_ylabel('Y坐标 (mm)')
        plt.colorbar(scatter, ax=axes[0, 1], label='Z坐标 (mm)')
        
        # 3. 深度分布
        axes[1, 0].hist(all_z, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_title('裂隙深度分布')
        axes[1, 0].set_xlabel('深度 (mm)')
        axes[1, 0].set_ylabel('裂隙数量')
        
        # 4. 钻孔布局
        for borehole_id, info in self.borehole_positions.items():
            axes[1, 1].plot(info['x'], info['y'], 'ro', markersize=10)
            axes[1, 1].text(info['x'], info['y'], borehole_id, 
                          ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        axes[1, 1].set_title('钻孔布局图')
        axes[1, 1].set_xlabel('X坐标 (mm)')
        axes[1, 1].set_ylabel('Y坐标 (mm)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_aspect('equal')
        
        plt.tight_layout()
        return fig

def main():
    """主函数"""
    print("="*60)
    print("🚀 Problem4 快速可视化程序")
    print("多钻孔裂隙网络三维可视化")
    print("="*60)
    
    # 初始化
    visualizer = QuickCrackVisualizer()
    
    # 1. 快速处理数据
    cracks_3d = visualizer.process_boreholes_quick(max_images_per_borehole=2)
    
    if not cracks_3d:
        print("❌ 无数据，退出")
        return
    
    # 2. 计算连通性
    connections = visualizer.calculate_simple_connectivity()
    
    # 3. 创建三维可视化
    fig_3d = visualizer.create_3d_plot(connections)
    
    # 保存交互式HTML
    os.makedirs("results", exist_ok=True)
    html_path = "results/quick_3d_visualization.html"
    fig_3d.write_html(html_path)
    print(f"✅ 交互式三维图已保存: {html_path}")
    
    # 4. 创建汇总图
    fig_summary = visualizer.create_summary_plot()
    summary_path = "results/quick_summary_analysis.png"
    fig_summary.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"✅ 汇总分析图已保存: {summary_path}")
    
    # 5. 统计信息
    total_cracks = sum(len(cracks) for cracks in cracks_3d.values())
    print(f"\n📊 统计结果:")
    print(f"  • 总裂隙数: {total_cracks}")
    print(f"  • 连通关系: {len(connections)}")
    print(f"  • 钻孔数: {len(cracks_3d)}")
    
    # 连通性统计
    high_prob = [c for c in connections if c['probability'] > 0.7]
    medium_prob = [c for c in connections if 0.4 <= c['probability'] <= 0.7]
    low_prob = [c for c in connections if 0.3 <= c['probability'] < 0.4]
    
    print(f"\n🔗 连通性分析:")
    print(f"  • 高概率连通 (>0.7): {len(high_prob)} 对")
    print(f"  • 中等概率连通 (0.4-0.7): {len(medium_prob)} 对")
    print(f"  • 低概率连通 (0.3-0.4): {len(low_prob)} 对")
    
    print(f"\n📁 输出文件:")
    print(f"  1. {html_path} - 交互式三维可视化")
    print(f"  2. {summary_path} - 汇总分析图")
    
    print("\n" + "="*60)
    print("🎉 快速可视化完成！")
    print("="*60)

if __name__ == "__main__":
    main()




