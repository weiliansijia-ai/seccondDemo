#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem4 演示可视化程序
展示多钻孔裂隙网络连通性分析的核心功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

class DemoCrackVisualizer:
    """演示版裂隙可视化器"""
    
    def __init__(self):
        """初始化"""
        # 钻孔布置参数 (2×3阵列)
        self.borehole_positions = {
            '1#': {'x': 500, 'y': 2000, 'z_start': 0, 'depth': 7000},
            '2#': {'x': 1500, 'y': 2000, 'z_start': 0, 'depth': 7000},
            '3#': {'x': 2500, 'y': 2000, 'z_start': 0, 'depth': 7000},
            '4#': {'x': 500, 'y': 1000, 'z_start': 0, 'depth': 5000},
            '5#': {'x': 1500, 'y': 1000, 'z_start': 0, 'depth': 7000},
            '6#': {'x': 2500, 'y': 1000, 'z_start': 0, 'depth': 7000},
        }
        
        self.bore_diameter = 30  # mm
        
    def generate_simulated_cracks(self):
        """生成模拟裂隙数据"""
        print("🔧 生成模拟裂隙数据...")
        
        np.random.seed(42)  # 固定随机种子
        simulated_cracks = {}
        
        for borehole_id, info in self.borehole_positions.items():
            cracks = []
            
            # 每个钻孔生成50-150条裂隙
            n_cracks = np.random.randint(50, 150)
            
            for i in range(n_cracks):
                # 随机生成裂隙位置
                depth = np.random.uniform(0, info['depth'])
                angle = np.random.uniform(0, 2*np.pi)
                radius = self.bore_diameter / 2
                
                # 钻孔壁面坐标
                x_rel = radius * np.cos(angle)
                y_rel = radius * np.sin(angle)
                
                # 绝对坐标
                x = info['x'] + x_rel
                y = info['y'] + y_rel
                z = info['z_start'] - depth
                
                # 裂隙属性
                crack = {
                    'x': x,
                    'y': y, 
                    'z': z,
                    'borehole_id': borehole_id,
                    'area': np.random.uniform(10, 200),
                    'length': np.random.uniform(5, 50),
                    'jrc': np.random.uniform(0, 15),  # JRC粗糙度
                    'inclination': np.random.uniform(-45, 45)  # 倾角
                }
                cracks.append(crack)
            
            simulated_cracks[borehole_id] = cracks
            print(f"  钻孔 {borehole_id}: 生成 {len(cracks)} 条裂隙")
        
        total = sum(len(cracks) for cracks in simulated_cracks.values())
        print(f"✅ 总共生成 {total} 条模拟裂隙")
        
        return simulated_cracks
    
    def calculate_connectivity(self, cracks_data):
        """计算连通性"""
        print("🔗 计算裂隙连通性...")
        
        connections = []
        borehole_ids = list(cracks_data.keys())
        
        # 计算钻孔间的连通关系
        for i, borehole_id1 in enumerate(borehole_ids):
            for j, borehole_id2 in enumerate(borehole_ids):
                if i >= j:
                    continue
                
                cracks1 = cracks_data[borehole_id1]
                cracks2 = cracks_data[borehole_id2]
                
                # 采样计算连通性（避免计算量过大）
                sample_size = min(20, len(cracks1), len(cracks2))
                indices1 = np.random.choice(len(cracks1), sample_size, replace=False)
                indices2 = np.random.choice(len(cracks2), sample_size, replace=False)
                
                for idx1 in indices1:
                    for idx2 in indices2:
                        crack1 = cracks1[idx1]
                        crack2 = cracks2[idx2]
                        
                        # 计算连通概率
                        prob = self._calculate_connectivity_probability(crack1, crack2)
                        
                        if prob > 0.2:  # 只保留概率>0.2的连通
                            connections.append({
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
                            })
        
        # 排序
        connections.sort(key=lambda x: x['probability'], reverse=True)
        
        # 分类
        high_prob = [c for c in connections if c['probability'] > 0.7]
        medium_prob = [c for c in connections if 0.5 <= c['probability'] <= 0.7]
        low_prob = [c for c in connections if 0.2 <= c['probability'] < 0.5]
        
        print(f"✅ 连通性分析完成:")
        print(f"  • 高概率连通 (>0.7): {len(high_prob)} 对")
        print(f"  • 中等概率连通 (0.5-0.7): {len(medium_prob)} 对")
        print(f"  • 低概率连通 (0.2-0.5): {len(low_prob)} 对")
        
        return {
            'all_connections': connections,
            'high_prob': high_prob,
            'medium_prob': medium_prob,
            'low_prob': low_prob
        }
    
    def _calculate_connectivity_probability(self, crack1, crack2):
        """计算两条裂隙的连通概率"""
        # 距离因子
        distance = np.sqrt(
            (crack1['x'] - crack2['x'])**2 + 
            (crack1['y'] - crack2['y'])**2 + 
            (crack1['z'] - crack2['z'])**2
        )
        distance_factor = max(0, 1 - distance / 2000)  # 2m最大连通距离
        
        # 方向因子
        angle_diff = abs(crack1['inclination'] - crack2['inclination'])
        angle_factor = max(0, 1 - angle_diff / 90)
        
        # 尺寸因子
        avg_size = (crack1['length'] + crack2['length']) / 2
        size_factor = min(1, avg_size / 30)
        
        # 粗糙度因子
        avg_jrc = (crack1['jrc'] + crack2['jrc']) / 2
        roughness_factor = max(0.1, 1 - avg_jrc / 20)
        
        # 深度对齐因子
        z_diff = abs(crack1['z'] - crack2['z'])
        depth_factor = max(0, 1 - z_diff / 1000)
        
        # 综合连通概率
        connectivity_prob = (
            distance_factor * 0.4 +
            angle_factor * 0.2 +
            size_factor * 0.15 +
            roughness_factor * 0.15 +
            depth_factor * 0.1
        )
        
        return max(0, min(1, connectivity_prob))
    
    def identify_uncertainty_regions(self, cracks_data):
        """识别不确定性区域"""
        print("🎯 识别高不确定性区域...")
        
        # 定义空间网格
        x_range = np.arange(0, 3000, 500)
        y_range = np.arange(500, 2500, 500)
        z_range = np.arange(-7000, 500, 1000)
        
        uncertainty_regions = []
        
        for i in range(len(x_range)-1):
            for j in range(len(y_range)-1):
                for k in range(len(z_range)-1):
                    x_center = (x_range[i] + x_range[i+1]) / 2
                    y_center = (y_range[j] + y_range[j+1]) / 2
                    z_center = (z_range[k] + z_range[k+1]) / 2
                    
                    # 计算该区域的观测密度
                    nearby_cracks = 0
                    for cracks in cracks_data.values():
                        for crack in cracks:
                            distance = np.sqrt(
                                (crack['x'] - x_center)**2 + 
                                (crack['y'] - y_center)**2 + 
                                (crack['z'] - z_center)**2
                            )
                            if distance < 750:
                                nearby_cracks += 1
                    
                    # 计算距离最近钻孔的距离
                    min_borehole_distance = min([
                        np.sqrt((info['x'] - x_center)**2 + (info['y'] - y_center)**2)
                        for info in self.borehole_positions.values()
                    ])
                    
                    # 计算不确定性
                    coverage = max(0, 1 - min_borehole_distance / 1500)
                    crack_density = min(1, nearby_cracks / 20)
                    uncertainty = (1 - coverage) * 0.7 + (1 - crack_density) * 0.3
                    
                    if uncertainty > 0.5:  # 高不确定性区域
                        uncertainty_regions.append({
                            'x': x_center,
                            'y': y_center,
                            'z': z_center,
                            'uncertainty_score': uncertainty,
                            'coverage_score': coverage
                        })
        
        # 排序
        uncertainty_regions.sort(key=lambda x: x['uncertainty_score'], reverse=True)
        
        print(f"✅ 识别出 {len(uncertainty_regions)} 个高不确定性区域")
        
        return uncertainty_regions[:15]  # 返回前15个
    
    def create_3d_visualization(self, cracks_data, connections, uncertainty_regions):
        """创建三维可视化"""
        print("🎨 创建三维可视化...")
        
        fig = go.Figure()
        
        # 1. 钻孔位置
        borehole_x = [info['x'] for info in self.borehole_positions.values()]
        borehole_y = [info['y'] for info in self.borehole_positions.values()]
        borehole_z = [info['z_start'] for info in self.borehole_positions.values()]
        borehole_names = list(self.borehole_positions.keys())
        
        fig.add_trace(go.Scatter3d(
            x=borehole_x, y=borehole_y, z=borehole_z,
            mode='markers+text',
            marker=dict(size=20, color='red', symbol='diamond'),
            text=borehole_names,
            textposition='top center',
            name='钻孔位置',
            hovertemplate='<b>钻孔 %{text}</b><br>坐标: (%{x}, %{y}, %{z})<extra></extra>'
        ))
        
        # 2. 裂隙点（按JRC着色）
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        for i, (borehole_id, cracks) in enumerate(cracks_data.items()):
            crack_x = [crack['x'] for crack in cracks]
            crack_y = [crack['y'] for crack in cracks]
            crack_z = [crack['z'] for crack in cracks]
            crack_jrc = [crack['jrc'] for crack in cracks]
            
            fig.add_trace(go.Scatter3d(
                x=crack_x, y=crack_y, z=crack_z,
                mode='markers',
                marker=dict(
                    size=4,
                    color=crack_jrc,
                    colorscale='Viridis',
                    colorbar=dict(title='JRC粗糙度', x=1.1) if i == 0 else None,
                    showscale=True if i == 0 else False,
                    opacity=0.7
                ),
                name=f'裂隙-{borehole_id}',
                hovertemplate=f'<b>钻孔 {borehole_id}</b><br>JRC: %{{marker.color:.1f}}<br>位置: (%{{x:.0f}}, %{{y:.0f}}, %{{z:.0f}})<extra></extra>'
            ))
        
        # 3. 高概率连通线
        high_prob_connections = connections['high_prob'][:15]  # 只显示前15个
        for i, conn in enumerate(high_prob_connections):
            fig.add_trace(go.Scatter3d(
                x=[conn['crack1']['x'], conn['crack2']['x']],
                y=[conn['crack1']['y'], conn['crack2']['y']],
                z=[conn['crack1']['z'], conn['crack2']['z']],
                mode='lines',
                line=dict(color='red', width=6),
                name='高概率连通' if i == 0 else None,
                showlegend=True if i == 0 else False,
                hovertemplate=f'<b>高概率连通</b><br>概率: {conn["probability"]:.3f}<br>距离: {conn["distance"]:.0f}mm<extra></extra>'
            ))
        
        # 4. 中等概率连通线
        medium_prob_connections = connections['medium_prob'][:10]
        for i, conn in enumerate(medium_prob_connections):
            fig.add_trace(go.Scatter3d(
                x=[conn['crack1']['x'], conn['crack2']['x']],
                y=[conn['crack1']['y'], conn['crack2']['y']],
                z=[conn['crack1']['z'], conn['crack2']['z']],
                mode='lines',
                line=dict(color='orange', width=4),
                name='中等概率连通' if i == 0 else None,
                showlegend=True if i == 0 else False,
                hovertemplate=f'<b>中等概率连通</b><br>概率: {conn["probability"]:.3f}<br>距离: {conn["distance"]:.0f}mm<extra></extra>'
            ))
        
        # 5. 不确定性区域
        if uncertainty_regions:
            uncertain_x = [region['x'] for region in uncertainty_regions]
            uncertain_y = [region['y'] for region in uncertainty_regions]
            uncertain_z = [region['z'] for region in uncertainty_regions]
            uncertain_scores = [region['uncertainty_score'] for region in uncertainty_regions]
            
            fig.add_trace(go.Scatter3d(
                x=uncertain_x, y=uncertain_y, z=uncertain_z,
                mode='markers',
                marker=dict(
                    size=15,
                    color=uncertain_scores,
                    colorscale='Reds',
                    opacity=0.8,
                    colorbar=dict(title='不确定性得分', x=1.2)
                ),
                name='高不确定性区域',
                hovertemplate='<b>不确定性区域</b><br>位置: (%{x:.0f}, %{y:.0f}, %{z:.0f})<br>不确定性: %{marker.color:.3f}<extra></extra>'
            ))
        
        # 设置布局
        fig.update_layout(
            title={
                'text': 'Problem4 - 多钻孔裂隙网络连通性三维分析<br><sub>Multi-Borehole Crack Network Connectivity Analysis</sub>',
                'x': 0.5,
                'font': dict(size=18)
            },
            scene=dict(
                xaxis_title='X坐标 (mm)',
                yaxis_title='Y坐标 (mm)',
                zaxis_title='Z坐标 (mm)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=1400,
            height=900,
            font=dict(family='Arial', size=12),
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def create_analysis_summary(self, cracks_data, connections, uncertainty_regions):
        """创建分析汇总图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Problem4 多钻孔裂隙网络连通性分析汇总', fontsize=16, fontweight='bold')
        
        # 1. 各钻孔裂隙数量
        borehole_names = list(cracks_data.keys())
        crack_counts = [len(cracks) for cracks in cracks_data.values()]
        
        bars1 = axes[0, 0].bar(borehole_names, crack_counts, color='steelblue', alpha=0.7)
        axes[0, 0].set_title('各钻孔检测到的裂隙数量')
        axes[0, 0].set_ylabel('裂隙数量')
        
        # 添加数值标签
        for bar, count in zip(bars1, crack_counts):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{count}', ha='center', va='bottom')
        
        # 2. 连通性统计
        conn_types = ['高概率', '中等概率', '低概率']
        conn_counts = [len(connections['high_prob']), len(connections['medium_prob']), len(connections['low_prob'])]
        colors_conn = ['red', 'orange', 'gray']
        
        bars2 = axes[0, 1].bar(conn_types, conn_counts, color=colors_conn, alpha=0.7)
        axes[0, 1].set_title('裂隙连通性分类统计')
        axes[0, 1].set_ylabel('连通对数')
        
        for bar, count in zip(bars2, conn_counts):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{count}', ha='center', va='bottom')
        
        # 3. 钻孔布局图
        for borehole_id, info in self.borehole_positions.items():
            axes[0, 2].plot(info['x'], info['y'], 'ro', markersize=12)
            axes[0, 2].text(info['x'], info['y'], borehole_id, 
                          ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        axes[0, 2].set_title('钻孔空间布局 (2×3阵列)')
        axes[0, 2].set_xlabel('X坐标 (mm)')
        axes[0, 2].set_ylabel('Y坐标 (mm)')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_aspect('equal')
        
        # 4. 裂隙空间分布
        all_x = []
        all_y = []
        all_z = []
        for cracks in cracks_data.values():
            all_x.extend([crack['x'] for crack in cracks])
            all_y.extend([crack['y'] for crack in cracks])
            all_z.extend([crack['z'] for crack in cracks])
        
        scatter = axes[1, 0].scatter(all_x, all_y, c=all_z, cmap='viridis', alpha=0.6, s=8)
        axes[1, 0].set_title('裂隙平面分布 (颜色表示深度)')
        axes[1, 0].set_xlabel('X坐标 (mm)')
        axes[1, 0].set_ylabel('Y坐标 (mm)')
        plt.colorbar(scatter, ax=axes[1, 0], label='Z坐标 (mm)')
        
        # 5. 连通概率分布
        all_probs = [conn['probability'] for conn in connections['all_connections']]
        if all_probs:
            axes[1, 1].hist(all_probs, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[1, 1].axvline(x=0.7, color='red', linestyle='--', label='高概率阈值')
            axes[1, 1].axvline(x=0.5, color='orange', linestyle='--', label='中等概率阈值')
            axes[1, 1].set_title('连通概率分布')
            axes[1, 1].set_xlabel('连通概率')
            axes[1, 1].set_ylabel('频数')
            axes[1, 1].legend()
        
        # 6. 不确定性区域分布
        if uncertainty_regions:
            uncertain_x = [region['x'] for region in uncertainty_regions]
            uncertain_y = [region['y'] for region in uncertainty_regions]
            uncertain_scores = [region['uncertainty_score'] for region in uncertainty_regions]
            
            scatter2 = axes[1, 2].scatter(uncertain_x, uncertain_y, c=uncertain_scores, 
                                        cmap='Reds', s=100, alpha=0.8, edgecolors='black')
            axes[1, 2].set_title('高不确定性区域分布')
            axes[1, 2].set_xlabel('X坐标 (mm)')
            axes[1, 2].set_ylabel('Y坐标 (mm)')
            plt.colorbar(scatter2, ax=axes[1, 2], label='不确定性得分')
        
        plt.tight_layout()
        return fig

def main():
    """主函数"""
    print("="*70)
    print("🎯 Problem4 演示可视化程序")
    print("多钻孔裂隙网络连通性分析与三维重构")
    print("="*70)
    
    # 初始化
    visualizer = DemoCrackVisualizer()
    
    # 1. 生成模拟数据
    cracks_data = visualizer.generate_simulated_cracks()
    
    # 2. 计算连通性
    connections = visualizer.calculate_connectivity(cracks_data)
    
    # 3. 识别不确定性区域
    uncertainty_regions = visualizer.identify_uncertainty_regions(cracks_data)
    
    # 4. 创建三维可视化
    fig_3d = visualizer.create_3d_visualization(cracks_data, connections, uncertainty_regions)
    
    # 5. 创建汇总分析图
    fig_summary = visualizer.create_analysis_summary(cracks_data, connections, uncertainty_regions)
    
    # 6. 保存结果
    os.makedirs("results", exist_ok=True)
    
    html_path = "results/demo_3d_connectivity.html"
    fig_3d.write_html(html_path)
    print(f"✅ 交互式三维可视化已保存: {html_path}")
    
    summary_path = "results/demo_analysis_summary.png"
    fig_summary.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"✅ 分析汇总图已保存: {summary_path}")
    
    # 7. 统计报告
    total_cracks = sum(len(cracks) for cracks in cracks_data.values())
    total_connections = len(connections['all_connections'])
    
    print(f"\n📊 分析结果统计:")
    print(f"  • 总裂隙数: {total_cracks}")
    print(f"  • 总连通关系: {total_connections}")
    print(f"  • 高概率连通: {len(connections['high_prob'])} 对")
    print(f"  • 中等概率连通: {len(connections['medium_prob'])} 对")
    print(f"  • 低概率连通: {len(connections['low_prob'])} 对")
    print(f"  • 高不确定性区域: {len(uncertainty_regions)} 个")
    
    print(f"\n📁 输出文件:")
    print(f"  1. {html_path} - 交互式三维可视化")
    print(f"  2. {summary_path} - 分析汇总图")
    
    print("\n💡 使用说明:")
    print("  • 打开HTML文件可进行三维交互操作")
    print("  • 红线表示高概率连通，橙线表示中等概率连通")
    print("  • 红色点表示高不确定性区域，需要补充观测")
    print("  • 钻孔按2×3阵列布置，间距1000mm")
    
    print("\n" + "="*70)
    print("🎉 Problem4 演示可视化完成！")
    print("="*70)

if __name__ == "__main__":
    main()




