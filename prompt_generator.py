#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据驱动信息图生成实验 - 提示词生成模块

基于ChartGalaxy方法论，实现模块化提示词模板设计和动态变量注入。
为27个实验条件生成精确的文本到图像生成指令。

作者: lxd
日期: 2025
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import anthropic
except ImportError:
    anthropic = None

class PromptGenerator:
    """智能提示词生成器，负责为不同的实验条件生成相应的提示词"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.llm_config = self.config.get('llm_config', {})
        self.intelligent_mode = self.config.get('enable_intelligent_processing', True)
        # 基础提示词模板
        self.base_template = """
Generate a clean, professional, and visually appealing infographic.

**Topic:**
"{topic}"

**Core Visualization:**
- Use a **{chart_type}** to visualize the following data: {data_string}.
- Ensure all data labels and corresponding values are clearly, accurately, and legibly represented in the chart. The visual representation (e.g., bar height, pie sector angle) must be proportional to the data values.

**Layout and Composition:**
- Strictly adhere to the following spatial arrangement of elements: **{layout_description}**.
- The overall style should be minimalist, modern, and uncluttered, with ample white space.

**Supplementary Elements:**
- Include the following brief descriptive text, positioned logically within the layout: "{text_description}".
- Subtly integrate one or two high-quality, vector-style icons related to these keywords: **{icon_keywords}**.

**Aesthetic Style:**
- Color Palette: Use a harmonious and professional color palette. For example, a primary color of deep blue (#003366) with analogous shades and a single accent color like gold (#FFD700). Avoid overly saturated or distracting colors.
- Typography: Use a clean, sans-serif font like Helvetica or Arial. Establish a clear visual hierarchy with distinct sizes and weights for the main title, subtitles, and body text.
""".strip()
        
        # 布局描述映射
        self.layout_descriptions = {
            "LT-01": "The main title is centered at the top. The chart is centered in the middle of the canvas. A descriptive text block is placed below the chart.",
            "LT-08": "The title is positioned in the top-left corner. The right half of the canvas is occupied by the chart. The left half of the canvas features a large, thematic icon.",
            "LT-25": "The chart itself serves as a full-bleed visual background. The title and descriptive text are overlaid on top of the chart area, typically with a semi-transparent background or drop shadow to ensure readability."
        }
    
    def serialize_data(self, data: List[Dict[str, Any]]) -> str:
        """将数据序列化为自然语言描述"""
        if not data:
            return "no data available"
        
        # 构建数据描述
        data_pairs = []
        for item in data:
            category = item.get('category', 'Unknown')
            value = item.get('value', 0)
            data_pairs.append(f"{category}: {value}")
        
        return f"a dataset with categories and values as follows: ({', '.join(data_pairs)})"
    
    def format_icon_keywords(self, keywords: List[str]) -> str:
        """格式化图标关键词"""
        if not keywords:
            return "data, chart, analysis"
        return ", ".join(keywords[:4])  # 限制为4个关键词
    
    def generate_single_prompt(self, 
                             data_sample: Dict[str, Any],
                             chart_type: str,
                             layout_template_id: str,
                             layout_template_name: str,
                             chart_description: str,
                             layout_description: str) -> str:
        """智能生成单个实验条件的提示词"""
        
        try:
            # 如果启用智能模式，尝试使用LLM优化提示词
            if self.intelligent_mode and self.llm_config.get('api_key'):
                enhanced_prompt = self._generate_enhanced_prompt_with_llm(
                    data_sample, chart_type, layout_template_id, layout_template_name, chart_description, layout_description
                )
                if enhanced_prompt:
                    return enhanced_prompt
            
            # 备用方案：使用规则生成
            return self._generate_rule_based_prompt(data_sample, chart_type, layout_template_id, layout_template_name, chart_description, layout_description)
            
        except Exception as e:
            print(f"生成提示词时发生错误: {e}")
            return self._generate_fallback_prompt(data_sample, chart_type, layout_template_id)
    
    def _generate_enhanced_prompt_with_llm(self, 
                                          data_sample: Dict[str, Any], 
                                          chart_type: str, 
                                          layout_template_id: str,
                                          layout_template_name: str,
                                          chart_description: str,
                                          layout_description: str) -> Optional[str]:
        """使用LLM生成增强的提示词"""
        try:
            # 构建LLM输入
            context = self._build_llm_context(data_sample, chart_type, layout_template_id, layout_template_name, chart_description, layout_description)
            
            prompt_for_llm = f"""
            #请根据以下信息生成一个专业的信息图生成提示词,生成的提示词将用于辅助ai生成符合要求的信息图：
            ##上下文:
            {context}
            ##上下文结束

            ###要求：
            1. 提示词应该清晰、具体、可执行
            2. 包含所有必要的设计元素和数据要求
            3. 符合指定的图表类型和布局要求
            4. 长度控制在200-300字之间
            5. 提示词能够帮助ai生成符合要求的信息图
            6. 使用英文编写
            
            ###生成示例:
            Generate a professional infographic with the title "Desktop Windows Version Market Share Worldwide". The central theme is the "Market share of Windows versions", which should be visually represented using a Pie Chart. The overall layout must strictly adhere to the LT-01 (Classic Centered Layout) template. The infographic needs to clearly display data for two time points: 2015 and 2016, with a focus on the change or comparison between these two years.
            Data Requirements:
            Primary Data Field: Version (used for categories/slices).
            Numerical Data Fields: 2015 (Primary value) and 2016 (Secondary value for comparison/context).
            The chart should visually show the market share distribution. Although the original data has 4 rows, the visual representation should be condensed or focused on the most significant 2 data points/versions (as suggested by the "数据量: 2条" field, implying a focus on the top 2 shares or a simplified view).
            Ensure the design is professional, clean, and modern. Include a legend or labels for the "Version" categories and clearly indicate the percentages for both 2015 and 2016 within the pie chart or accompanying text/tables. The total data volume is based on 4 original rows, but the presentation should be simplified, perhaps showing the two largest segments and "Others." The final infographic should be compelling and easy to read.
            Keywords: data visualization, technology, software market, operating system, corporate blue palette, modern design.

            #请直接返回提示词内容，不要其他解释。
            """
            
            response = self._call_llm_api(prompt_for_llm)
            if response and len(response.strip()) > 50:
                return response.strip()
                
        except Exception as e:
            print(f"LLM提示词生成失败: {e}")
        
        return None
    
    def _generate_rule_based_prompt(self, 
                                   data_sample: Dict[str, Any], 
                                   chart_type: str, 
                                   layout_template_id: str,
                                   layout_template_name: str,
                                   chart_description: str,
                                   layout_description: str) -> str:
        """基于规则生成提示词"""
        # 从新的数据结构中提取信息
        metadata = data_sample.get('metadata', {})
        topic = metadata.get('title', 'Data Analysis')
        text_description = metadata.get('main_insight', 'Data visualization showing key insights.')
        
        # 从data部分获取实际数据
        data_info = data_sample.get('data', {})
        actual_data = data_info.get('data', [])
        
        # 序列化数据
        data_string = self.serialize_data(actual_data)
        
        # 生成图标关键词（基于数据主题）
        icon_keywords = ['data', 'chart', 'analysis']
        if 'windows' in topic.lower():
            icon_keywords.extend(['windows', 'computer', 'system'])
        elif 'market' in topic.lower():
            icon_keywords.extend(['market', 'business', 'share'])
        
        # 格式化图标关键词
        formatted_keywords = self.format_icon_keywords(icon_keywords)
        
        # 填充模板
        prompt = self.base_template.format(
            topic=topic,
            chart_type=chart_type,
            data_string=data_string,
            layout_description=layout_description,
            text_description=text_description,
            icon_keywords=formatted_keywords
        )
        
        return prompt
    
    def _generate_fallback_prompt(self, 
                                 data_sample: Dict[str, Any], 
                                 chart_type: str, 
                                 layout_template_id: str,
                                 layout_template_name: str,
                                 chart_description: str,
                                 layout_description: str) -> str:
        """生成备用提示词"""
        metadata = data_sample.get('metadata', {})
        topic = metadata.get('title', 'data analysis')
        return f"Generate a {chart_type} chart for {topic} using {layout_template_id} ({layout_template_name}) layout. {chart_description}"
    
    def _build_llm_context(self, 
                          data_sample: Dict[str, Any], 
                          chart_type: str, 
                          layout_template_id: str,
                          layout_template_name: str,
                          chart_description: str,
                          layout_description: str) -> str:
        """构建LLM上下文"""
        context_parts = []
        
        # 从metadata获取数据信息
        metadata = data_sample.get('metadata', {})
        context_parts.append(f"数据主题: {metadata.get('title', '数据分析')}")
        context_parts.append(f"数据描述: {metadata.get('description', '')}")
        context_parts.append(f"主要洞察: {metadata.get('main_insight', '')}")
        
        # 从data部分获取数据特征
        data_info = data_sample.get('data', {})
        columns_info = data_info.get('columns', [])
        actual_data = data_info.get('data', [])
        
        if actual_data:
            context_parts.append(f"数据量: {len(actual_data)}条")
        
        if columns_info:
            # 提取列名和重要性信息
            primary_columns = [col['name'] for col in columns_info if col.get('importance') == 'primary']
            secondary_columns = [col['name'] for col in columns_info if col.get('importance') == 'secondary']
            
            if primary_columns:
                context_parts.append(f"主要字段: {', '.join(primary_columns)}")
            if secondary_columns:
                context_parts.append(f"次要字段: {', '.join(secondary_columns)}")
            
            # 数据类型信息
            categorical_cols = [col['name'] for col in columns_info if col.get('data_type') == 'categorical']
            numerical_cols = [col['name'] for col in columns_info if col.get('data_type') == 'numerical']
            
            if categorical_cols:
                context_parts.append(f"分类字段: {', '.join(categorical_cols)}")
            if numerical_cols:
                context_parts.append(f"数值字段: {', '.join(numerical_cols)}")
        
        # 设计要求
        context_parts.append(f"图表类型: {chart_type}")
        context_parts.append(f"图表描述: {chart_description}")
        context_parts.append(f"布局模板: {layout_template_id} ({layout_template_name})")
        context_parts.append(f"布局描述: {layout_description}")
        
        # 处理信息
        processing_info = metadata.get('processing_info', {})
        if processing_info:
            context_parts.append(f"样本ID: {processing_info.get('sample_id', '')}")
            context_parts.append(f"原始数据行数: {processing_info.get('original_rows', '')}")
        
        return "\n".join(context_parts)
    
    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """调用LLM API"""
        api_type = self.llm_config.get('api_type', 'openai')
        api_key = self.llm_config.get('api_key')
        
        if not api_key:
            return None
        
        try:
            if api_type == 'openai':
                return self._call_openai_api(prompt, api_key)
            elif api_type == 'claude':
                return self._call_claude_api(prompt, api_key)
        except Exception as e:
            print(f"LLM API调用失败: {e}")
        
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
                temperature=0.7
            )
            
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
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Claude API调用失败: {e}")
            return None
    
    def load_data_samples(self, data_dir: str = ".") -> Dict[str, Dict[str, Any]]:
        """加载所有数据样本"""
        data_samples = {}
        data_path = Path(data_dir)
        
        # 加载3个数据样本文件
        for i in range(1, 4):
            filename = f"data_sample_{i:02d}.json"
            filepath = data_path / filename
            
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data_samples[f"DS-{i:02d}"] = json.load(f)
            else:
                # 创建默认样本
                data_samples[f"DS-{i:02d}"] = self._create_default_sample(i)
        
        return data_samples
    
    def _create_default_sample(self, sample_num: int) -> Dict[str, Any]:
        """创建默认数据样本"""
        default_data = [
            {"category": "Q1", "value": 120 + sample_num * 10},
            {"category": "Q2", "value": 150 + sample_num * 15},
            {"category": "Q3", "value": 135 + sample_num * 12},
            {"category": "Q4", "value": 180 + sample_num * 20}
        ]
        
        return {
            "metadata": {
                "title": f"季度业绩分析 - 样本{sample_num}",
                "description": "展示季度数据变化趋势的分析图表",
                "main_insight": f"数据显示第四季度表现最佳，实现了{sample_num * 5}%的增长。",
                "processing_info": {
                    "sample_id": f"{sample_num}",
                    "processing_timestamp": "2024-01-01T00:00:00.000000",
                    "original_rows": 4,
                    "processing_method": "default_generation"
                }
            },
            "data": {
                "columns": [
                    {
                        "name": "category",
                        "importance": "primary",
                        "description": "季度分类",
                        "unit": "none",
                        "data_type": "categorical",
                        "role": "x"
                    },
                    {
                        "name": "value",
                        "importance": "primary",
                        "description": "业绩数值",
                        "unit": "none",
                        "data_type": "numerical",
                        "role": "y"
                    }
                ],
                "data": default_data
            }
        }
    
    def generate_all_prompts(self, 
                           matrix_file: str = "experimental_matrix.csv",
                           data_dir: str = ".") -> List[Dict[str, Any]]:
        """生成所有27个实验条件的提示词"""
        
        # 加载实验矩阵
        matrix = pd.read_csv(matrix_file)
        
        # 加载数据样本
        data_samples = self.load_data_samples(data_dir)
        
        # 生成所有提示词
        all_prompts = []
        
        for _, row in matrix.iterrows():
            run_id = row['Run ID']
            data_sample_id = row['Data Sample ID']
            chart_type = row['Chart Type']
            layout_template_id = row['Layout Template ID']
            layout_template_name = row['Layout Name']
            chart_description = row['Chart Description']
            layout_description = row['Layout Description']
            # 获取对应的数据样本
            data_sample = data_samples.get(data_sample_id, self._create_default_sample(1))
            
            # 生成提示词
            prompt = self.generate_single_prompt(
                data_sample=data_sample,
                chart_type=chart_type,
                layout_template_id=layout_template_id,
                layout_template_name=layout_template_name,
                chart_description=chart_description,
                layout_description=layout_description
            )
            
            # 记录提示词信息
            prompt_info = {
                "run_id": run_id,
                "data_sample_id": data_sample_id,
                "chart_type": chart_type,
                "layout_template_id": layout_template_id,
                "prompt": prompt,
                "data": data_sample,
                "metadata": {
                    "topic": data_sample.get('topic', ''),
                    "data_points": len(data_sample.get('data', [])),
                    "prompt_length": len(prompt)
                }
            }
            
            all_prompts.append(prompt_info)
        
        return all_prompts
    
    def save_prompts(self, 
                    prompts: List[Dict[str, Any]], 
                    output_file: str = "prompts.txt",
                    json_output: str = "all_prompts.json") -> tuple:
        """保存提示词到文件"""
        
        # 保存简单文本格式（每行一个提示词）
        with open(output_file, 'w', encoding='utf-8') as f:
            for prompt_info in prompts:
                f.write(f"# {prompt_info['run_id']} - {prompt_info['chart_type']} - {prompt_info['layout_template_id']}\n")
                f.write(prompt_info['prompt'])
                f.write("\n\n" + "="*80 + "\n\n")
        
        # 保存详细JSON格式
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        
        return output_file, json_output
    
    def validate_prompts(self, prompts: List[Dict[str, Any]]) -> bool:
        """验证生成的提示词"""
        print("验证提示词...")
        
        # 检查数量
        if len(prompts) != 27:
            print(f"错误: 提示词数量不正确。期望: 27, 实际: {len(prompts)}")
            return False
        
        # 检查每个提示词的完整性
        for i, prompt_info in enumerate(prompts):
            prompt = prompt_info['prompt']
            
            # 检查是否包含所有必要的占位符内容
            required_sections = ['Topic:', 'Core Visualization:', 'Layout and Composition:', 
                               'Supplementary Elements:', 'Aesthetic Style:']
            
            for section in required_sections:
                if section not in prompt:
                    print(f"错误: 提示词 {i+1} 缺少必要部分: {section}")
                    return False
            
            # 检查是否还有未替换的占位符
            if '{' in prompt or '}' in prompt:
                print(f"警告: 提示词 {i+1} 可能包含未替换的占位符")
        
        print("✓ 提示词验证通过")
        return True
    
    def print_generation_summary(self, prompts: List[Dict[str, Any]]):
        """打印生成摘要"""
        print("=" * 60)
        print("提示词生成摘要")
        print("=" * 60)
        print(f"总提示词数量: {len(prompts)}")
        
        # 统计各类型数量
        chart_types = {}
        layout_types = {}
        
        for prompt_info in prompts:
            chart_type = prompt_info['chart_type']
            layout_type = prompt_info['layout_template_id']
            
            chart_types[chart_type] = chart_types.get(chart_type, 0) + 1
            layout_types[layout_type] = layout_types.get(layout_type, 0) + 1
        
        print("\n图表类型分布:")
        for chart_type, count in chart_types.items():
            print(f"  {chart_type}: {count}")
        
        print("\n布局模板分布:")
        for layout_type, count in layout_types.items():
            print(f"  {layout_type}: {count}")
        
        # 提示词长度统计
        lengths = [prompt_info['metadata']['prompt_length'] for prompt_info in prompts]
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)
        
        print(f"\n提示词长度统计:")
        print(f"  平均长度: {avg_length:.0f} 字符")
        print(f"  最短长度: {min_length} 字符")
        print(f"  最长长度: {max_length} 字符")
        
        print("=" * 60)

def main():
    """主函数"""
    print("开始生成提示词...")
    
    # 加载配置（如果存在）
    config = {}
    config_file = Path("config.json")
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 创建提示词生成器
    generator = PromptGenerator(config)
    
    try:
        # 生成所有提示词
        prompts = generator.generate_all_prompts()
        
        # 验证提示词
        if generator.validate_prompts(prompts):
            # 保存提示词
            text_file, json_file = generator.save_prompts(prompts)
            
            # 打印摘要
            generator.print_generation_summary(prompts)
            
            print(f"\n提示词已保存到:")
            print(f"  文本格式: {text_file}")
            print(f"  JSON格式: {json_file}")
            
            print("\n提示词生成完成！")
        else:
            print("提示词验证失败，请检查配置")
            
    except FileNotFoundError as e:
        print(f"错误: 找不到必要的文件 - {e}")
        print("请确保已运行数据预处理和实验矩阵生成脚本")
    except Exception as e:
        print(f"生成提示词时发生错误: {e}")

if __name__ == "__main__":
    main()