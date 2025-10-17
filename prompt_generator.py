#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据驱动信息图生成实验 - 提示词生成模块

基于ChartGalaxy方法论，实现模块化提示词模板设计和动态变量注入。
为27个实验条件生成精确的文本到图像生成指令。

作者: ChartGalaxy Pipeline
日期: 2024
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
                             layout_template_id: str) -> str:
        """智能生成单个实验条件的提示词"""
        
        try:
            # 如果启用智能模式，尝试使用LLM优化提示词
            if self.intelligent_mode and self.llm_config.get('api_key'):
                enhanced_prompt = self._generate_enhanced_prompt_with_llm(
                    data_sample, chart_type, layout_template_id
                )
                if enhanced_prompt:
                    return enhanced_prompt
            
            # 备用方案：使用规则生成
            return self._generate_rule_based_prompt(data_sample, chart_type, layout_template_id)
            
        except Exception as e:
            print(f"生成提示词时发生错误: {e}")
            return self._generate_fallback_prompt(data_sample, chart_type, layout_template_id)
    
    def _generate_enhanced_prompt_with_llm(self, 
                                          data_sample: Dict[str, Any], 
                                          chart_type: str, 
                                          layout_template_id: str) -> Optional[str]:
        """使用LLM生成增强的提示词"""
        try:
            # 构建LLM输入
            context = self._build_llm_context(data_sample, chart_type, layout_template_id)
            
            prompt_for_llm = f"""
            请根据以下信息生成一个专业的信息图生成提示词：
            
            {context}
            
            要求：
            1. 提示词应该清晰、具体、可执行
            2. 包含所有必要的设计元素和数据要求
            3. 符合指定的图表类型和布局要求
            4. 长度控制在200-300字之间
            
            请直接返回提示词内容，不要其他解释。
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
                                   layout_template_id: str) -> str:
        """基于规则生成提示词"""
        # 提取数据样本信息
        topic = data_sample.get('topic', 'Data Analysis')
        data = data_sample.get('data', [])
        elements = data_sample.get('elements', {})
        text_description = elements.get('text', 'Data visualization showing key insights.')
        icon_keywords = elements.get('icon_keywords', ['data', 'chart'])
        
        # 序列化数据
        data_string = self.serialize_data(data)
        
        # 获取布局描述
        layout_description = self.layout_descriptions.get(
            layout_template_id, 
            "Standard layout with title at top and chart in center."
        )
        
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
                                 layout_template_id: str) -> str:
        """生成备用提示词"""
        return f"Generate a {chart_type} chart for {data_sample.get('topic', 'data analysis')} using {layout_template_id} layout."
    
    def _build_llm_context(self, 
                          data_sample: Dict[str, Any], 
                          chart_type: str, 
                          layout_template_id: str) -> str:
        """构建LLM上下文"""
        context_parts = []
        
        # 数据信息
        context_parts.append(f"主题: {data_sample.get('topic', '数据分析')}")
        context_parts.append(f"用户需求: {data_sample.get('query', '')}")
        
        # 数据特征
        data = data_sample.get('data', [])
        if data:
            context_parts.append(f"数据量: {len(data)}条")
            if data:
                columns = list(data[0].keys())
                context_parts.append(f"数据字段: {', '.join(columns)}")
        
        # 设计要求
        context_parts.append(f"图表类型: {chart_type}")
        context_parts.append(f"布局模板: {layout_template_id}")
        
        # 补充元素
        elements = data_sample.get('elements', {})
        if elements.get('text'):
            context_parts.append(f"关键洞察: {elements['text']}")
        
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
                max_tokens=500,
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
            "sample_id": f"DS-{sample_num:02d}",
            "topic": f"季度业绩分析 - 样本{sample_num}",
            "query": "创建一个展示季度数据的信息图",
            "data": default_data,
            "elements": {
                "text": f"数据显示第四季度表现最佳，实现了{sample_num * 5}%的增长。",
                "icon_keywords": ["business", "growth", "chart", "analysis"]
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
            
            # 获取对应的数据样本
            data_sample = data_samples.get(data_sample_id, self._create_default_sample(1))
            
            # 生成提示词
            prompt = self.generate_single_prompt(
                data_sample=data_sample,
                chart_type=chart_type,
                layout_template_id=layout_template_id
            )
            
            # 记录提示词信息
            prompt_info = {
                "run_id": run_id,
                "data_sample_id": data_sample_id,
                "chart_type": chart_type,
                "layout_template_id": layout_template_id,
                "prompt": prompt,
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
                    json_output: str = "prompts_detailed.json") -> tuple:
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