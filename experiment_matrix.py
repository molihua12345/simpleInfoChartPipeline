#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据驱动信息图生成实验 - 实验矩阵生成模块

基于ChartGalaxy分类体系，生成3×3×3的全因子实验设计矩阵。
包含3个数据样本、3种图表类型、3种布局模板的所有组合。

作者: lxd
日期: 2025
"""

import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path

class ExperimentMatrix:
    """实验矩阵生成器"""
    
    def __init__(self):
        # 定义图表类型（基于ChartGalaxy分类体系）
        self.chart_types = {
            "Vertical Bar Chart": {
                "name": "Vertical Bar Chart",
                "description": "垂直条形图，用于基础的类别间数值比较",
                "best_for": "categorical data comparison"
            },
            "Pie Chart": {
                "name": "Pie Chart", 
                "description": "饼图，用于展示整体构成中的各部分占比",
                "best_for": "part-to-whole relationships"
            },
            "Line Graph": {
                "name": "Line Graph",
                "description": "折线图，用于展示数据随时间或连续维度的变化趋势",
                "best_for": "trends and changes over time"
            }
        }
        
        # 定义布局模板（基于ChartGalaxy布局分类）
        self.layout_templates = {
            "LT-01": {
                "id": "LT-01",
                "name": "Classic Centered Layout",
                "description": "标题顶部居中，图表主体居中，文字描述位于图表下方",
                "spatial_arrangement": "The main title is centered at the top. The chart is centered in the middle of the canvas. A descriptive text block is placed below the chart."
            },
            "LT-08": {
                "id": "LT-08", 
                "name": "Asymmetric Split Layout",
                "description": "标题位于左上角，图表主体占据右侧区域，大型主题图标占据左侧区域",
                "spatial_arrangement": "The title is positioned in the top-left corner. The right half of the canvas is occupied by the chart. The left half of the canvas features a large, thematic icon."
            },
            "LT-25": {
                "id": "LT-25",
                "name": "Immersive Overlay Layout", 
                "description": "图表作为视觉背景占据大部分版面，标题和文字以叠加形式放置",
                "spatial_arrangement": "The chart itself serves as a full-bleed visual background. The title and descriptive text are overlaid on top of the chart area, typically with a semi-transparent background or drop shadow to ensure readability."
            }
        }
        
        # 定义数据样本ID
        self.data_samples = ["DS-01", "DS-02", "DS-03"]
    
    def generate_matrix(self) -> pd.DataFrame:
        """生成完整的3×3×3实验矩阵"""
        matrix_data = []
        run_id = 1
        
        for data_sample in self.data_samples:
            for chart_type in self.chart_types.keys():
                for layout_template in self.layout_templates.keys():
                    matrix_data.append({
                        "Run ID": f"EXP-{run_id:02d}",
                        "Data Sample ID": data_sample,
                        "Chart Type": chart_type,
                        "Layout Template ID": layout_template,
                        "Layout Name": self.layout_templates[layout_template]["name"],
                        "Chart Description": self.chart_types[chart_type]["description"],
                        "Layout Description": self.layout_templates[layout_template]["description"]
                    })
                    run_id += 1
        
        return pd.DataFrame(matrix_data)
    
    def get_chart_type_info(self, chart_type: str) -> Dict:
        """获取图表类型详细信息"""
        return self.chart_types.get(chart_type, {})
    
    def get_layout_template_info(self, layout_id: str) -> Dict:
        """获取布局模板详细信息"""
        return self.layout_templates.get(layout_id, {})
    
    def save_matrix(self, output_path: str = "experimental_matrix.csv") -> str:
        """保存实验矩阵到CSV文件"""
        matrix = self.generate_matrix()
        matrix.to_csv(output_path, index=False, encoding='utf-8-sig')
        return output_path
    
    def print_matrix_summary(self):
        """打印实验矩阵摘要"""
        matrix = self.generate_matrix()
        
        print("=" * 60)
        print("实验设计矩阵摘要")
        print("=" * 60)
        print(f"总实验次数: {len(matrix)}")
        print(f"数据样本数: {len(self.data_samples)}")
        print(f"图表类型数: {len(self.chart_types)}")
        print(f"布局模板数: {len(self.layout_templates)}")
        print()
        
        print("图表类型:")
        for i, (chart_type, info) in enumerate(self.chart_types.items(), 1):
            print(f"  {i}. {chart_type}: {info['description']}")
        print()
        
        print("布局模板:")
        for i, (layout_id, info) in enumerate(self.layout_templates.items(), 1):
            print(f"  {i}. {layout_id} ({info['name']}): {info['description']}")
        print()
        
        print("实验矩阵前5行预览:")
        print(matrix[['Run ID', 'Data Sample ID', 'Chart Type', 'Layout Template ID']].head())
        print("=" * 60)
    
    def validate_matrix(self) -> bool:
        """验证实验矩阵的完整性"""
        matrix = self.generate_matrix()
        
        # 检查总数是否正确
        expected_total = len(self.data_samples) * len(self.chart_types) * len(self.layout_templates)
        if len(matrix) != expected_total:
            print(f"错误: 实验矩阵总数不正确。期望: {expected_total}, 实际: {len(matrix)}")
            return False
        
        # 检查每个组合是否唯一
        combinations = matrix[['Data Sample ID', 'Chart Type', 'Layout Template ID']]
        if len(combinations.drop_duplicates()) != len(combinations):
            print("错误: 发现重复的实验组合")
            return False
        
        # 检查Run ID是否连续
        run_ids = [int(rid.split('-')[1]) for rid in matrix['Run ID']]
        if run_ids != list(range(1, len(matrix) + 1)):
            print("错误: Run ID不连续")
            return False
        
        print("✓ 实验矩阵验证通过")
        return True

def main():
    """主函数"""
    print("生成实验设计矩阵...")
    
    # 创建实验矩阵生成器
    matrix_generator = ExperimentMatrix()
    
    # 打印摘要
    matrix_generator.print_matrix_summary()
    
    # 验证矩阵
    if matrix_generator.validate_matrix():
        # 保存矩阵
        output_file = matrix_generator.save_matrix()
        print(f"\n实验矩阵已保存到: {output_file}")
        
        # 生成详细的实验配置文件
        matrix = matrix_generator.generate_matrix()
        
        # 保存详细配置
        detailed_config = {
            "experiment_info": {
                "total_experiments": len(matrix),
                "data_samples": len(matrix_generator.data_samples),
                "chart_types": len(matrix_generator.chart_types),
                "layout_templates": len(matrix_generator.layout_templates)
            },
            "chart_types": matrix_generator.chart_types,
            "layout_templates": matrix_generator.layout_templates,
            "experiment_matrix": matrix.to_dict('records')
        }
        
        import json
        with open('experiment_config.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_config, f, ensure_ascii=False, indent=2)
        
        print("详细实验配置已保存到: experiment_config.json")
    else:
        print("实验矩阵验证失败，请检查配置")

if __name__ == "__main__":
    main()