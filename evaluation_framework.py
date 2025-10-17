#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据驱动信息图生成实验 - 评估框架模块

基于ChartGalaxy评估维度，实现多维度量化评估框架。
包含数据一致性、布局准确性和美观度三个核心评估维度。

作者: ChartGalaxy Pipeline
日期: 2024
"""

import json
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# 条件导入LLM库
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

class EvaluationDimension(Enum):
    """评估维度枚举"""
    DATA_CONSISTENCY = "data_consistency"
    LAYOUT_ACCURACY = "layout_accuracy"
    AESTHETIC_QUALITY = "aesthetic_quality"

@dataclass
class EvaluationCriteria:
    """评估标准数据类"""
    dimension: EvaluationDimension
    name: str
    description: str
    max_score: int
    criteria: Dict[str, str]

class EvaluationFramework:
    """智能评估框架类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.llm_config = self.config.get('llm_config', {})
        self.intelligent_mode = self.config.get('enable_intelligent_processing', True)
        self.evaluation_criteria = self._define_evaluation_criteria()
        self.scoring_rubric = self._create_scoring_rubric()
        self.evaluation_history = []
    
    def _define_evaluation_criteria(self) -> Dict[EvaluationDimension, EvaluationCriteria]:
        """定义评估标准"""
        criteria = {}
        
        # 1. 数据一致性评估标准
        criteria[EvaluationDimension.DATA_CONSISTENCY] = EvaluationCriteria(
            dimension=EvaluationDimension.DATA_CONSISTENCY,
            name="数据一致性",
            description="评估生成图像中的数据表示与原始数据的匹配程度",
            max_score=10,
            criteria={
                "数值准确性": "图表中显示的数值与原始数据是否完全匹配",
                "类别标签": "类别名称是否正确显示且清晰可读",
                "比例关系": "视觉元素（如条形高度、饼图扇形角度）是否与数据成比例",
                "数据完整性": "是否包含了所有应该显示的数据点"
            }
        )
        
        # 2. 布局准确性评估标准
        criteria[EvaluationDimension.LAYOUT_ACCURACY] = EvaluationCriteria(
            dimension=EvaluationDimension.LAYOUT_ACCURACY,
            name="布局准确性",
            description="评估生成图像的布局是否符合指定的布局模板要求",
            max_score=10,
            criteria={
                "元素位置": "标题、图表、文本等元素是否按照布局模板正确放置",
                "空间分配": "各元素占用的空间比例是否合理",
                "对齐方式": "元素的对齐方式是否符合布局要求",
                "层次结构": "视觉层次是否清晰，重要元素是否突出"
            }
        )
        
        # 3. 美观度评估标准
        criteria[EvaluationDimension.AESTHETIC_QUALITY] = EvaluationCriteria(
            dimension=EvaluationDimension.AESTHETIC_QUALITY,
            name="美观度",
            description="评估生成图像的整体视觉质量和设计美感",
            max_score=10,
            criteria={
                "色彩搭配": "颜色选择是否和谐，对比度是否适当",
                "字体排版": "字体选择和排版是否专业、易读",
                "视觉平衡": "整体构图是否平衡，视觉重心是否合适",
                "设计一致性": "整体设计风格是否统一，是否符合专业信息图标准"
            }
        )
        
        return criteria
    
    def _create_scoring_rubric(self) -> Dict[str, Dict[int, str]]:
        """创建评分标准表"""
        return {
            "data_consistency": {
                10: "完美 - 所有数据完全准确，无任何错误",
                8: "优秀 - 数据基本准确，仅有微小偏差",
                6: "良好 - 数据大体正确，有少量明显错误",
                4: "一般 - 数据部分正确，有多处错误",
                2: "较差 - 数据错误较多，但仍可识别",
                0: "极差 - 数据完全错误或无法识别"
            },
            "layout_accuracy": {
                10: "完美 - 完全符合布局模板要求",
                8: "优秀 - 基本符合布局要求，仅有微小偏差",
                6: "良好 - 大体符合布局，有少量位置偏差",
                4: "一般 - 部分符合布局，有明显偏差",
                2: "较差 - 布局偏差较大，但仍可识别模板特征",
                0: "极差 - 完全不符合布局要求"
            },
            "aesthetic_quality": {
                10: "完美 - 专业级设计质量，视觉效果卓越",
                8: "优秀 - 设计质量很高，视觉效果良好",
                6: "良好 - 设计合理，视觉效果可接受",
                4: "一般 - 设计基本可用，但缺乏美感",
                2: "较差 - 设计质量低，视觉效果不佳",
                0: "极差 - 设计质量极差，难以接受"
            }
        }
    
    def create_evaluation_template(self) -> Dict[str, Any]:
        """创建评估模板"""
        template = {
            "experiment_info": {
                "run_id": "",
                "data_sample_id": "",
                "chart_type": "",
                "layout_template_id": "",
                "evaluator": "",
                "evaluation_date": "",
                "image_path": ""
            },
            "scores": {},
            "detailed_feedback": {},
            "overall_score": 0,
            "notes": ""
        }
        
        # 为每个评估维度创建评分字段
        for dimension, criteria in self.evaluation_criteria.items():
            dim_key = dimension.value
            template["scores"][dim_key] = {
                "total_score": 0,
                "max_score": criteria.max_score,
                "sub_scores": {}
            }
            
            # 为每个子标准创建评分字段
            for sub_criterion in criteria.criteria.keys():
                template["scores"][dim_key]["sub_scores"][sub_criterion] = 0
            
            template["detailed_feedback"][dim_key] = ""
        
        return template
    
    def generate_evaluation_batch(self, 
                                matrix_file: str = "experimental_matrix.csv",
                                output_dir: str = "evaluations") -> str:
        """生成批量评估模板"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 加载实验矩阵
        matrix = pd.read_csv(matrix_file)
        
        # 为每个实验生成评估模板
        evaluation_templates = []
        
        for _, row in matrix.iterrows():
            template = self.create_evaluation_template()
            
            # 填充实验信息
            template["experiment_info"].update({
                "run_id": row['Run ID'],
                "data_sample_id": row['Data Sample ID'],
                "chart_type": row['Chart Type'],
                "layout_template_id": row['Layout Template ID'],
                "image_path": f"generated_images/{row['Run ID']}.png"
            })
            
            evaluation_templates.append(template)
        
        # 保存批量评估模板
        batch_file = output_path / "evaluation_batch.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_templates, f, ensure_ascii=False, indent=2)
        
        # 生成评估指南
        self._generate_evaluation_guide(output_path)
        
        return str(batch_file)
    
    def evaluate_infographic(self, 
                           infographic_data: Dict[str, Any], 
                           generated_image_path: Optional[str] = None) -> Dict[str, Any]:
        """智能评估单个信息图"""
        try:
            # 如果启用智能模式且有LLM配置，使用AI辅助评估
            if self.intelligent_mode and self.llm_config.get('api_key'):
                ai_evaluation = self._perform_ai_assisted_evaluation(
                    infographic_data, generated_image_path
                )
                if ai_evaluation:
                    return ai_evaluation
            
            # 备用方案：使用规则评估
            return self._perform_rule_based_evaluation(infographic_data, generated_image_path)
            
        except Exception as e:
            print(f"评估信息图时发生错误: {e}")
            return self._create_default_evaluation(infographic_data)
    
    def _perform_ai_assisted_evaluation(self, 
                                       infographic_data: Dict[str, Any], 
                                       generated_image_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """执行AI辅助评估"""
        try:
            # 构建评估上下文
            context = self._build_evaluation_context(infographic_data, generated_image_path)
            
            # 生成评估提示词
            evaluation_prompt = self._create_evaluation_prompt(context)
            
            # 调用LLM进行评估
            llm_response = self._call_llm_for_evaluation(evaluation_prompt)
            
            if llm_response:
                # 解析LLM响应并结合规则评估
                return self._combine_ai_and_rule_evaluation(
                    llm_response, infographic_data, generated_image_path
                )
        except Exception as e:
            print(f"AI辅助评估失败: {e}")
        
        return None
    
    def _perform_rule_based_evaluation(self, 
                                      infographic_data: Dict[str, Any], 
                                      generated_image_path: Optional[str] = None) -> Dict[str, Any]:
        """执行基于规则的评估"""
        evaluation_result = {
            'infographic_id': infographic_data.get('sample_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'scores': {},
            'overall_score': 0.0,
            'detailed_feedback': {},
            'recommendations': [],
            'evaluation_method': 'rule_based'
        }
        
        # 基于现有评估标准进行评分
        total_score = 0
        max_total_score = 0
        
        for dimension, criteria in self.evaluation_criteria.items():
            # 简化的规则评估逻辑
            score = self._evaluate_dimension_by_rules(dimension, infographic_data)
            evaluation_result['scores'][dimension.value] = {
                'score': score,
                'max_score': criteria.max_score
            }
            
            total_score += score
            max_total_score += criteria.max_score
        
        evaluation_result['overall_score'] = (total_score / max_total_score * 10) if max_total_score > 0 else 0.0
        
        # 记录评估历史
        self.evaluation_history.append(evaluation_result)
        
        return evaluation_result
    
    def _evaluate_dimension_by_rules(self, dimension: EvaluationDimension, infographic_data: Dict[str, Any]) -> int:
        """基于规则评估单个维度"""
        # 简化的规则评估逻辑
        if dimension == EvaluationDimension.DATA_CONSISTENCY:
            # 检查数据完整性
            data = infographic_data.get('data', [])
            if data and len(data) > 0:
                return 8  # 有数据给8分
            return 5  # 无数据给5分
        
        elif dimension == EvaluationDimension.LAYOUT_ACCURACY:
            # 检查布局信息
            layout_id = infographic_data.get('layout_template_id', '')
            if layout_id:
                return 7  # 有布局模板给7分
            return 5  # 无布局模板给5分
        
        elif dimension == EvaluationDimension.AESTHETIC_QUALITY:
            # 基于图表类型评估美观度
            chart_type = infographic_data.get('chart_type', '')
            if chart_type in ['bar_chart', 'pie_chart', 'line_chart']:
                return 7  # 常见图表类型给7分
            return 6  # 其他类型给6分
        
        return 6  # 默认分数
    
    def _build_evaluation_context(self, 
                                 infographic_data: Dict[str, Any], 
                                 generated_image_path: Optional[str] = None) -> Dict[str, Any]:
        """构建评估上下文"""
        context = {
            'topic': infographic_data.get('topic', ''),
            'query': infographic_data.get('query', ''),
            'data_summary': self._summarize_data(infographic_data.get('data', [])),
            'chart_type': infographic_data.get('chart_type', ''),
            'elements': infographic_data.get('elements', {}),
            'has_image': generated_image_path is not None,
            'metadata': infographic_data.get('metadata', {})
        }
        return context
    
    def _create_evaluation_prompt(self, context: Dict[str, Any]) -> str:
        """创建评估提示词"""
        prompt = f"""
        请对以下信息图设计进行专业评估，从以下维度给出1-10分的评分和具体建议：
        
        信息图信息：
        - 主题：{context['topic']}
        - 用户需求：{context['query']}
        - 数据概况：{context['data_summary']}
        - 图表类型：{context['chart_type']}
        
        评估维度：
        1. 数据一致性 (1-10分)：数据表示是否准确、完整
        2. 布局准确性 (1-10分)：布局是否合理、美观
        3. 美学质量 (1-10分)：视觉设计是否专业、吸引人
        
        请按以下格式返回评估结果：
        数据一致性: [分数] - [简短评价]
        布局准确性: [分数] - [简短评价]
        美学质量: [分数] - [简短评价]
        总体建议: [改进建议]
        """
        return prompt
    
    def _call_llm_for_evaluation(self, prompt: str) -> Optional[str]:
        """调用LLM进行评估"""
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
            print(f"LLM评估API调用失败: {e}")
        
        return None
    
    def _call_openai_api(self, prompt: str, api_key: str) -> Optional[str]:
        """调用OpenAI API"""
        if OpenAI is None:
            print("openai库未安装，无法调用OpenAI API")
            return None
            
        try:
            # 创建OpenAI客户端
            client_kwargs = {"api_key": api_key}
            
            # 添加base_url和timeout配置
            if 'base_url' in self.llm_config:
                client_kwargs['base_url'] = self.llm_config['base_url']
            if 'timeout' in self.llm_config:
                client_kwargs['timeout'] = self.llm_config['timeout']
            
            client = OpenAI(**client_kwargs)
            
            # 调用API
            response = client.chat.completions.create(
                model=self.llm_config.get('model', 'gpt-3.5-turbo'),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
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
            # 创建Anthropic客户端
            client_kwargs = {"api_key": api_key}
            
            # 添加timeout配置
            if 'timeout' in self.llm_config:
                client_kwargs['timeout'] = self.llm_config['timeout']
            
            client = anthropic.Anthropic(**client_kwargs)
            
            # 调用API
            response = client.messages.create(
                model=self.llm_config.get('model', 'claude-3-haiku-20240307'),
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Claude API调用失败: {e}")
            return None
    
    def _combine_ai_and_rule_evaluation(self, 
                                       llm_response: str, 
                                       infographic_data: Dict[str, Any], 
                                       generated_image_path: Optional[str] = None) -> Dict[str, Any]:
        """结合AI和规则评估结果"""
        # 解析LLM响应
        ai_scores = self._parse_llm_evaluation(llm_response)
        
        # 执行规则评估
        rule_evaluation = self._perform_rule_based_evaluation(infographic_data, generated_image_path)
        
        # 结合两种评估结果
        combined_evaluation = {
            'infographic_id': infographic_data.get('sample_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'scores': {},
            'overall_score': 0.0,
            'detailed_feedback': {},
            'recommendations': [],
            'evaluation_method': 'ai_assisted',
            'ai_feedback': llm_response,
            'rule_scores': rule_evaluation['scores']
        }
        
        # 加权平均AI和规则评分
        ai_weight = 0.7  # AI评估权重
        rule_weight = 0.3  # 规则评估权重
        
        total_score = 0.0
        
        for dimension in self.evaluation_criteria.keys():
            dim_key = dimension.value
            ai_score = ai_scores.get(dim_key, 6.0)
            rule_score_data = rule_evaluation['scores'].get(dim_key, {'score': 6})
            rule_score = rule_score_data['score'] if isinstance(rule_score_data, dict) else rule_score_data
            
            # 加权平均
            combined_score = (ai_score * ai_weight + rule_score * rule_weight)
            combined_evaluation['scores'][dim_key] = {
                'score': combined_score,
                'ai_score': ai_score,
                'rule_score': rule_score
            }
            
            total_score += combined_score
        
        # 计算总分
        combined_evaluation['overall_score'] = total_score / len(self.evaluation_criteria) if self.evaluation_criteria else 0.0
        
        # 生成综合建议
        combined_evaluation['recommendations'] = self._generate_enhanced_recommendations(
            combined_evaluation, llm_response
        )
        
        # 记录评估历史
        self.evaluation_history.append(combined_evaluation)
        
        return combined_evaluation
    
    def _parse_llm_evaluation(self, llm_response: str) -> Dict[str, float]:
        """解析LLM评估响应"""
        scores = {}
        
        # 映射LLM维度到系统维度
        dimension_mapping = {
            '数据一致性': 'data_consistency',
            '布局准确性': 'layout_accuracy', 
            '美学质量': 'aesthetic_quality'
        }
        
        import re
        
        for chinese_name, english_name in dimension_mapping.items():
            # 查找评分模式
            pattern = rf'{chinese_name}:\s*(\d+(?:\.\d+)?)'
            match = re.search(pattern, llm_response)
            if match:
                try:
                    score = float(match.group(1))
                    scores[english_name] = max(1.0, min(10.0, score))  # 限制在1-10分范围内
                except ValueError:
                    scores[english_name] = 6.0  # 默认分数
            else:
                scores[english_name] = 6.0  # 默认分数
        
        return scores
    
    def _generate_enhanced_recommendations(self, 
                                         evaluation_result: Dict[str, Any], 
                                         ai_feedback: str) -> List[str]:
        """生成增强的改进建议"""
        recommendations = []
        
        # 基于分数的建议
        for dim_key, score_data in evaluation_result['scores'].items():
            score = score_data['score'] if isinstance(score_data, dict) else score_data
            if score < 6.0:
                dimension_name = {
                    'data_consistency': '数据一致性',
                    'layout_accuracy': '布局准确性',
                    'aesthetic_quality': '美学质量'
                }.get(dim_key, dim_key)
                recommendations.append(f"改进{dimension_name}：当前得分{score:.1f}，需要重点关注")
        
        # 从AI反馈中提取建议
        if '总体建议:' in ai_feedback:
            ai_suggestions = ai_feedback.split('总体建议:')[-1].strip()
            if ai_suggestions:
                recommendations.append(f"AI建议：{ai_suggestions}")
        
        # 如果没有具体建议，提供通用建议
        if not recommendations:
            overall_score = evaluation_result['overall_score']
            if overall_score < 5.0:
                recommendations.append("整体表现需要改进，建议重新审视设计方案")
            elif overall_score < 7.0:
                recommendations.append("表现良好，可在细节方面进一步优化")
            else:
                recommendations.append("表现优秀，保持当前设计水准")
        
        return recommendations[:5]  # 限制建议数量
    
    def _summarize_data(self, data: List[Dict]) -> str:
        """总结数据信息"""
        if not data:
            return "无数据"
        
        summary_parts = []
        summary_parts.append(f"数据量: {len(data)}条")
        
        if data:
            columns = list(data[0].keys())
            summary_parts.append(f"字段: {', '.join(columns[:3])}{'...' if len(columns) > 3 else ''}")
        
        return "; ".join(summary_parts)
    
    def _create_default_evaluation(self, infographic_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建默认评估结果"""
        return {
            'infographic_id': infographic_data.get('sample_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'scores': {
                'data_consistency': {'score': 5, 'max_score': 10},
                'layout_accuracy': {'score': 5, 'max_score': 10},
                'aesthetic_quality': {'score': 5, 'max_score': 10}
            },
            'overall_score': 5.0,
            'detailed_feedback': {
                'data_consistency': '评估失败，使用默认分数',
                'layout_accuracy': '评估失败，使用默认分数',
                'aesthetic_quality': '评估失败，使用默认分数'
            },
            'recommendations': ['评估过程中出现错误，建议重新评估'],
            'evaluation_method': 'default',
            'error': True
        }
    
    def _generate_evaluation_guide(self, output_path: Path):
        """生成评估指南文档"""
        guide_content = """
# 信息图生成质量评估指南

## 评估概述

本评估框架基于ChartGalaxy方法论，从三个核心维度对生成的信息图进行量化评估：

1. **数据一致性** (Data Consistency) - 最高10分
2. **布局准确性** (Layout Accuracy) - 最高10分  
3. **美观度** (Aesthetic Quality) - 最高10分

总分：30分

## 详细评估标准

### 1. 数据一致性 (10分)

评估生成图像中的数据表示与原始数据的匹配程度。

**子标准：**
- **数值准确性** (2.5分)：图表中显示的数值与原始数据是否完全匹配
- **类别标签** (2.5分)：类别名称是否正确显示且清晰可读
- **比例关系** (2.5分)：视觉元素（如条形高度、饼图扇形角度）是否与数据成比例
- **数据完整性** (2.5分)：是否包含了所有应该显示的数据点

**评分标准：**
- 10分：完美 - 所有数据完全准确，无任何错误
- 8分：优秀 - 数据基本准确，仅有微小偏差
- 6分：良好 - 数据大体正确，有少量明显错误
- 4分：一般 - 数据部分正确，有多处错误
- 2分：较差 - 数据错误较多，但仍可识别
- 0分：极差 - 数据完全错误或无法识别

### 2. 布局准确性 (10分)

评估生成图像的布局是否符合指定的布局模板要求。

**布局模板说明：**
- **LT-01 (Classic Centered Layout)**: 标题顶部居中，图表主体居中，文字描述位于图表下方
- **LT-08 (Asymmetric Split Layout)**: 标题位于左上角，图表占据右侧，大型图标占据左侧
- **LT-25 (Immersive Overlay Layout)**: 图表作为背景，标题和文字叠加显示

**子标准：**
- **元素位置** (2.5分)：标题、图表、文本等元素是否按照布局模板正确放置
- **空间分配** (2.5分)：各元素占用的空间比例是否合理
- **对齐方式** (2.5分)：元素的对齐方式是否符合布局要求
- **层次结构** (2.5分)：视觉层次是否清晰，重要元素是否突出

### 3. 美观度 (10分)

评估生成图像的整体视觉质量和设计美感。

**子标准：**
- **色彩搭配** (2.5分)：颜色选择是否和谐，对比度是否适当
- **字体排版** (2.5分)：字体选择和排版是否专业、易读
- **视觉平衡** (2.5分)：整体构图是否平衡，视觉重心是否合适
- **设计一致性** (2.5分)：整体设计风格是否统一，是否符合专业信息图标准

## 评估流程

1. 打开对应的生成图像文件
2. 查看原始数据和提示词要求
3. 按照三个维度逐一评分
4. 为每个子标准给出具体分数
5. 计算各维度总分
6. 填写详细反馈意见
7. 记录整体评价和改进建议

## 注意事项

- 评估时请保持客观公正
- 每个维度的评分应基于具体的评估标准
- 建议多次查看图像以确保评估准确性
- 详细记录发现的问题和亮点
- 如有疑问，可参考ChartGalaxy论文中的评估案例
"""
        
        guide_file = output_path / "evaluation_guide.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
    
    def calculate_scores(self, evaluation_data: Dict[str, Any]) -> Dict[str, float]:
        """计算评估分数"""
        scores = evaluation_data.get('scores', {})
        results = {}
        
        total_score = 0
        max_total_score = 0
        
        for dimension in self.evaluation_criteria.keys():
            dim_key = dimension.value
            if dim_key in scores:
                dim_score = scores[dim_key].get('total_score', 0)
                dim_max = scores[dim_key].get('max_score', 10)
                
                results[dim_key] = {
                    'score': dim_score,
                    'max_score': dim_max,
                    'percentage': (dim_score / dim_max * 100) if dim_max > 0 else 0
                }
                
                total_score += dim_score
                max_total_score += dim_max
        
        results['overall'] = {
            'score': total_score,
            'max_score': max_total_score,
            'percentage': (total_score / max_total_score * 100) if max_total_score > 0 else 0
        }
        
        return results
    
    def analyze_evaluation_results(self, evaluations_dir: str = "evaluations") -> pd.DataFrame:
        """分析评估结果"""
        evaluations_path = Path(evaluations_dir)
        
        # 查找所有评估结果文件
        evaluation_files = list(evaluations_path.glob("*_evaluation.json"))
        
        if not evaluation_files:
            print("未找到评估结果文件")
            return pd.DataFrame()
        
        # 收集所有评估数据
        results_data = []
        
        for eval_file in evaluation_files:
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            # 计算分数
            scores = self.calculate_scores(eval_data)
            
            # 提取实验信息
            exp_info = eval_data.get('experiment_info', {})
            
            result_row = {
                'run_id': exp_info.get('run_id', ''),
                'data_sample_id': exp_info.get('data_sample_id', ''),
                'chart_type': exp_info.get('chart_type', ''),
                'layout_template_id': exp_info.get('layout_template_id', ''),
                'data_consistency_score': scores.get('data_consistency', {}).get('score', 0),
                'layout_accuracy_score': scores.get('layout_accuracy', {}).get('score', 0),
                'aesthetic_quality_score': scores.get('aesthetic_quality', {}).get('score', 0),
                'overall_score': scores.get('overall', {}).get('score', 0),
                'overall_percentage': scores.get('overall', {}).get('percentage', 0)
            }
            
            results_data.append(result_row)
        
        return pd.DataFrame(results_data)
    
    def generate_evaluation_report(self, 
                                 results_df: pd.DataFrame, 
                                 output_file: str = "evaluation_report.md") -> str:
        """生成评估报告"""
        if results_df.empty:
            return "无评估数据可生成报告"
        
        # 计算统计数据
        stats = {
            'total_experiments': len(results_df),
            'avg_overall_score': results_df['overall_score'].mean(),
            'avg_data_consistency': results_df['data_consistency_score'].mean(),
            'avg_layout_accuracy': results_df['layout_accuracy_score'].mean(),
            'avg_aesthetic_quality': results_df['aesthetic_quality_score'].mean(),
            'best_performing': results_df.loc[results_df['overall_score'].idxmax()],
            'worst_performing': results_df.loc[results_df['overall_score'].idxmin()]
        }
        
        # 按图表类型分组统计
        chart_type_stats = results_df.groupby('chart_type').agg({
            'overall_score': ['mean', 'std', 'count'],
            'data_consistency_score': 'mean',
            'layout_accuracy_score': 'mean',
            'aesthetic_quality_score': 'mean'
        }).round(2)
        
        # 按布局模板分组统计
        layout_stats = results_df.groupby('layout_template_id').agg({
            'overall_score': ['mean', 'std', 'count'],
            'data_consistency_score': 'mean',
            'layout_accuracy_score': 'mean',
            'aesthetic_quality_score': 'mean'
        }).round(2)
        
        # 生成报告内容
        report_content = f"""
# 信息图生成质量评估报告

## 评估概览

- **总实验数量**: {stats['total_experiments']}
- **平均总分**: {stats['avg_overall_score']:.2f}/30 ({stats['avg_overall_score']/30*100:.1f}%)
- **平均数据一致性得分**: {stats['avg_data_consistency']:.2f}/10
- **平均布局准确性得分**: {stats['avg_layout_accuracy']:.2f}/10
- **平均美观度得分**: {stats['avg_aesthetic_quality']:.2f}/10

## 最佳和最差表现

### 最佳表现实验
- **实验ID**: {stats['best_performing']['run_id']}
- **图表类型**: {stats['best_performing']['chart_type']}
- **布局模板**: {stats['best_performing']['layout_template_id']}
- **总分**: {stats['best_performing']['overall_score']:.2f}/30

### 最差表现实验
- **实验ID**: {stats['worst_performing']['run_id']}
- **图表类型**: {stats['worst_performing']['chart_type']}
- **布局模板**: {stats['worst_performing']['layout_template_id']}
- **总分**: {stats['worst_performing']['overall_score']:.2f}/30

## 按图表类型分析

{chart_type_stats.to_string()}

## 按布局模板分析

{layout_stats.to_string()}

## 详细结果

{results_df.to_string(index=False)}

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return output_file

def main():
    """主函数"""
    print("初始化评估框架...")
    
    # 创建评估框架
    framework = EvaluationFramework()
    
    try:
        # 生成评估模板
        batch_file = framework.generate_evaluation_batch()
        print(f"评估模板已生成: {batch_file}")
        
        # 打印评估标准
        print("\n=" * 60)
        print("评估标准概览")
        print("=" * 60)
        
        for dimension, criteria in framework.evaluation_criteria.items():
            print(f"\n{criteria.name} (最高{criteria.max_score}分):")
            print(f"  {criteria.description}")
            for sub_name, sub_desc in criteria.criteria.items():
                print(f"  - {sub_name}: {sub_desc}")
        
        print("\n评估框架初始化完成！")
        print("\n下一步：")
        print("1. 运行图像生成流程")
        print("2. 使用evaluation_guide.md进行人工评估")
        print("3. 运行结果分析生成报告")
        
    except Exception as e:
        print(f"初始化评估框架时发生错误: {e}")

if __name__ == "__main__":
    main()