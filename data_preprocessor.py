#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据驱动信息图生成实验 - 数据预处理模块

基于MatPlotBench数据集，实现从原始数据到结构化JSON的转换流程。
按照ChartGalaxy方法论进行数据抽样、主题提取和补充元素生成。

作者: lxd
日期: 2025
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import random
from pathlib import Path
from datetime import datetime
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import anthropic
except ImportError:
    anthropic = None

class DataPreprocessor:
    """数据预处理器，负责将MatPlotBench数据转换为实验所需的结构化格式"""
    
    def __init__(self, benchmark_data_path: str, config: Optional[Dict] = None):
        self.benchmark_data_path = Path(benchmark_data_path)
        self.data_path = self.benchmark_data_path / "data"
        self.instructions_path = self.benchmark_data_path / "benchmark_instructions.json"
        self.config = config or {}
        self.llm_config = self.config.get('llm_config', {})
        self.random_seed = self.config.get('random_seed', 42)
        random.seed(self.random_seed)
        
        # 加载指令数据
        with open(self.instructions_path, 'r', encoding='utf-8') as f:
            self.instructions = json.load(f)
    
    def load_sample_data(self, sample_id: str) -> Tuple[pd.DataFrame, str]:
        """加载指定样本的数据和查询指令，如果CSV超过10行则截断"""
        sample_dir = self.data_path / sample_id
        
        # 查找数据文件
        data_file = None
        for ext in ['.csv', '.json', '.tsv']:
            potential_file = sample_dir / f"data{ext}"
            if potential_file.exists():
                data_file = potential_file
                break
        
        if data_file is None:
            raise FileNotFoundError(f"No data file found in {sample_dir}")
        
        # 加载数据
        if data_file.suffix == '.csv':
            data = pd.read_csv(data_file)
            # 如果数据行超过10行，截断保留表头和前10行
            if len(data) > 10:
                data = data.head(10)
                print(f"CSV数据超过10行，已截断为前10行数据 (样本: {sample_id})")
        elif data_file.suffix == '.json':
            with open(data_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            # 简化JSON数据处理，提取主要数值数据
            data = self._extract_data_from_json(json_data)
        else:
            data = pd.read_csv(data_file, sep='\t')
            # 对TSV文件也应用同样的截断逻辑
            if len(data) > 10:
                data = data.head(10)
                print(f"TSV数据超过10行，已截断为前10行数据 (样本: {sample_id})")
        
        # 获取对应的查询指令
        instruction = self._get_instruction_by_id(int(sample_id))
        
        return data, instruction
    
    def _extract_data_from_json(self, json_data: Dict) -> pd.DataFrame:
        """从复杂JSON结构中提取可用于可视化的数据"""
        # 这里实现一个简化的JSON数据提取逻辑
        # 实际应用中需要根据具体的JSON结构进行调整
        if 'data' in json_data and isinstance(json_data['data'], list):
            # 尝试提取第一个数据项的结构
            first_item = json_data['data'][0] if json_data['data'] else {}
            if 'node' in first_item and 'label' in first_item['node']:
                # Sankey图数据结构
                labels = first_item['node']['label'][:6]  # 取前6个标签
                values = list(range(len(labels)))  # 生成示例数值
                return pd.DataFrame({'category': labels, 'value': values})
        
        # 默认返回示例数据
        return pd.DataFrame({
            'category': ['A', 'B', 'C', 'D'],
            'value': [10, 20, 15, 25]
        })
    
    def _get_instruction_by_id(self, sample_id: int) -> str:
        """根据样本ID获取对应的查询指令"""
        for instruction in self.instructions:
            if instruction.get('id') == sample_id:
                return instruction.get('simple_instruction', '')
        return "Generate a data visualization chart."
    
    def convert_dataframe_to_json(self, data: pd.DataFrame, instruction: str) -> Dict[str, Any]:
        """将DataFrame转换为符合task2.md规范的JSON格式"""
        # 生成列定义
        columns_def = []
        data_records = []
        
        # 分析每一列的特征
        for i, col_name in enumerate(data.columns):
            col_data = data[col_name]
            
            # 判断数据类型
            if pd.api.types.is_numeric_dtype(col_data):
                data_type = "numerical"
                unit = "none"
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                data_type = "temporal"
                unit = "none"
            else:
                data_type = "categorical"
                unit = "none"
            
            # 判断重要性（第一列通常是primary，其他为secondary）
            importance = "primary" if i < 2 else "secondary"
            
            # 判断在可视化中的角色
            if i == 0:
                role = "x"
            elif data_type == "numerical":
                role = "y"
            else:
                role = "group"
            
            columns_def.append({
                "name": col_name,
                "importance": importance,
                "description": f"Data column representing {col_name.lower()}",
                "unit": unit,
                "data_type": data_type,
                "role": role
            })
        
        # 转换数据记录
        for _, row in data.iterrows():
            record = {}
            for col_name in data.columns:
                value = row[col_name]
                # 处理NaN值
                if pd.isna(value):
                    record[col_name] = None
                else:
                    record[col_name] = value
            data_records.append(record)
        
        # 构建完整的JSON结构
        result = {
            "metadata": {
                "title": "Data Visualization Chart",
                "description": f"Data visualization based on the instruction: {instruction[:100]}...",
                "main_insight": "This chart shows the relationship between different data points."
            },
            "data": {
                "columns": columns_def,
                "data": data_records
            }
        }
        
        return result
    
    def generate_ai_metadata(self, data: pd.DataFrame, instruction: str) -> Dict[str, str]:
        """使用AI生成更好的metadata信息"""
        # 创建数据摘要
        data_summary = self._create_dataframe_summary(data)
        
        prompt = f"""
你是一个数据分析专家。请根据以下数据和指令，生成合适的图表元数据。

数据摘要：
{data_summary}

用户指令：
{instruction}

请生成以下三个字段的内容（用JSON格式返回）：
1. title: 简洁有力的图表标题（不超过50字符）
2. description: 对数据内容的简要描述（不超过100字符）
3. main_insight: 数据揭示的主要洞察或趋势（不超过150字符）

返回格式：
{{
  "title": "图表标题",
  "description": "数据描述",
  "main_insight": "主要洞察"
}}
"""
        
        try:
            ai_response = self._call_llm_api(prompt)
            if ai_response:
                # 尝试解析AI返回的JSON
                import re
                json_match = re.search(r'\{[^}]+\}', ai_response, re.DOTALL)
                if json_match:
                    metadata = json.loads(json_match.group())
                    return {
                        "title": metadata.get("title", "Data Visualization Chart"),
                        "description": metadata.get("description", f"Visualization based on: {instruction[:80]}..."),
                        "main_insight": metadata.get("main_insight", "This chart reveals patterns in the data.")
                    }
                else:
                    print("AI返回的JSON格式错误")
            else:
                print("AI返回空响应")
        except Exception as e:
            print(f"AI metadata generation failed: {e}")
        
        # 回退到基础metadata
        return {
            "title": "Data Visualization Chart",
            "description": f"Visualization based on: {instruction[:80]}...",
            "main_insight": "This chart reveals patterns in the data."
        }
    
    def _create_dataframe_summary(self, data: pd.DataFrame) -> str:
        """创建DataFrame的摘要信息"""
        summary_parts = []
        summary_parts.append(f"数据行数: {len(data)}")
        summary_parts.append(f"数据列数: {len(data.columns)}")
        summary_parts.append(f"列名: {', '.join(data.columns.tolist())}")
        
        # 添加每列的基本统计信息
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                summary_parts.append(f"{col}: 数值型，范围 {data[col].min():.2f} - {data[col].max():.2f}")
            else:
                unique_count = data[col].nunique()
                summary_parts.append(f"{col}: 分类型，{unique_count}个不同值")
        
        return "\n".join(summary_parts)
    
    def process_sample(self, sample_id: str) -> Dict[str, Any]:
        """处理单个样本，生成符合规范的JSON结构"""
        try:
            # 加载原始数据（已包含截断逻辑）
            raw_data, instruction = self.load_sample_data(sample_id)
            
            # 将DataFrame转换为基础JSON格式
            json_data = self.convert_dataframe_to_json(raw_data, instruction)
            
            # 使用AI生成更好的metadata
            ai_metadata = self.generate_ai_metadata(raw_data, instruction)
            
            # 更新metadata
            json_data["metadata"].update(ai_metadata)
            
            # 添加处理信息
            json_data["metadata"]["processing_info"] = {
                "sample_id": sample_id,
                "processing_timestamp": datetime.now().isoformat(),
                "original_rows": len(raw_data),
                "processing_method": "data_process_pipeline"
            }
            
            return json_data
            
        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            # 返回基础的默认样本
            return {}
    
    
    def generate_experiment_samples(self, output_dir: str = ".") -> List[str]:
        """生成实验所需的3个数据样本"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 获取可用样本列表
        available_samples = self._get_available_samples()
        
        # 智能随机选择3个代表性样本ID
        sample_ids = self._select_representative_samples(available_samples, num_samples=3)
        generated_files = []
        
        for i, sample_id in enumerate(sample_ids, 1):
            # 处理样本
            sample_data = self.process_sample(sample_id)
            
            # 保存到文件
            filename = f"data_sample_{i:02d}.json"
            filepath = output_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
            generated_files.append(str(filepath))
            print(f"Generated sample {i}: {filename}")
        
        return generated_files
    
    def _get_available_samples(self) -> List[str]:
        """获取可用的样本ID列表"""
        if self.data_path.exists():
            # 从数据目录获取样本ID
            sample_dirs = [d.name for d in self.data_path.iterdir() if d.is_dir()]
            return sample_dirs[:20]  # 限制最多20个样本
        else:
            # 使用默认样本ID
            return [str(i) for i in range(1, 21)]
    
    def _select_representative_samples(self, available_samples: List[str], num_samples: int = 3) -> List[str]:
        """智能选择代表性样本"""
        if len(available_samples) <= num_samples:
            return available_samples
        
        # 如果样本数量足够，进行智能选择
        try:
            # 尝试均匀分布选择
            step = len(available_samples) // num_samples
            selected = []
            for i in range(num_samples):
                idx = i * step + random.randint(0, step // 2) if step > 1 else i
                if idx < len(available_samples):
                    selected.append(available_samples[idx])
            
            # 如果选择不足，随机补充
            while len(selected) < num_samples:
                remaining = [s for s in available_samples if s not in selected]
                if remaining:
                    selected.append(random.choice(remaining))
                else:
                    break
            
            return selected[:num_samples]
        except Exception:
            # 备用方案：完全随机选择
            return random.sample(available_samples, min(num_samples, len(available_samples)))
    
    
    
    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """调用LLM API"""
        api_type = self.llm_config.get('api_type', 'openai')
        api_key = self.llm_config.get('api_key')
        
        if not api_key:
            return None
        
        if api_type == 'openai':
            return self._call_openai_api(prompt, api_key)
        elif api_type == 'claude':
            return self._call_claude_api(prompt, api_key)
        else:
            return None
    
    def _call_openai_api(self, prompt: str, api_key: str) -> Optional[str]:
        """调用OpenAI API"""
        if OpenAI is None:
            print("OpenAI库未安装，无法调用OpenAI API")
            return None
        
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=self.llm_config.get('base_url', 'https://api.openai.com/v1'),
                timeout=self.llm_config.get('timeout', 30)
            )
            
            response = client.chat.completions.create(
                model=self.llm_config.get('model', 'gpt-3.5-turbo'),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            print(response.choices[0].message.content.strip())
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            return None
    
    def _call_claude_api(self, prompt: str, api_key: str) -> Optional[str]:
        """调用Claude API"""
        if anthropic is None:
            print("anthropic库未安装，无法调用Claude API")
            return None
        
        try:
            client = anthropic.Anthropic(
                api_key=api_key,
                timeout=self.llm_config.get('timeout', 30)
            )
            
            response = client.messages.create(
                model=self.llm_config.get('model', 'claude-3-haiku-20240307'),
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Claude API调用失败: {e}")
            return None

def main():
    """主函数"""
    # 设置路径
    benchmark_data_path = "benchmark_data"
    
    # 创建预处理器
    preprocessor = DataPreprocessor(benchmark_data_path)
    
    # 生成实验样本
    print("开始生成实验数据样本...")
    generated_files = preprocessor.generate_experiment_samples()
    
    print(f"\n成功生成 {len(generated_files)} 个数据样本:")
    for file in generated_files:
        print(f"  - {file}")
    
    print("\n数据预处理完成！")

if __name__ == "__main__":
    main()