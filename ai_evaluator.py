#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI驱动评估模块

使用多模态AI模型（如GPT-4V、Claude Vision等）自动评估生成的信息图质量。
基于ChartGalaxy评估维度进行量化评分和详细反馈生成。

作者: lxd
日期: 2025
"""

import os
import json
import base64
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from PIL import Image
import pandas as pd

# 条件导入LLM库
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

class AIEvaluatorType(Enum):
    """AI评估器类型枚举"""
    GPT4V = "gpt4v"
    CLAUDE_VISION = "claude_vision"
    GEMINI_VISION = "gemini_vision"

@dataclass
class EvaluationConfig:
    """评估配置"""
    evaluator_type: AIEvaluatorType
    api_key: str
    model: str = "gpt-4-vision-preview"
    max_tokens: int = 1500
    temperature: float = 0.1
    retry_attempts: int = 3
    retry_delay: int = 5

class AIEvaluator:
    """AI评估器类"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluation_results = []
        self.failed_evaluations = []
        
        # 评估提示词模板
        self.evaluation_prompt_template = self._create_evaluation_prompt_template()
    
    def _create_evaluation_prompt_template(self) -> str:
        """创建评估提示词模板"""
        template_file = os.path.join(os.path.dirname(__file__), "evaluation_prompt_template.txt")
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"警告：找不到提示词模板文件 {template_file}，使用默认模板")
            # 如果文件不存在，返回一个简化的默认模板
            raise FileNotFoundError(f"评估提示词模板文件 {template_file} 不存在")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图像编码为base64字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _extract_json_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        """从AI响应中提取JSON内容"""
        import re
        
        # 方法1: 尝试提取markdown代码块中的JSON
        json_pattern = r'```(?:json)?\s*({[^`]+})\s*```'
        match = re.search(json_pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 方法2: 查找完整的JSON对象
        brace_count = 0
        start_idx = -1
        
        for i, char in enumerate(content):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        json_content = content[start_idx:i+1]
                        return json.loads(json_content)
                    except json.JSONDecodeError:
                        continue
        
        # 方法3: 简单的首尾查找（原始方法的备用）
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            try:
                json_content = content[json_start:json_end]
                return json.loads(json_content)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def evaluate_single_image(self, 
                            image_path: str, 
                            experiment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """评估单张图像"""
        
        if not Path(image_path).exists():
            print(f"图像文件不存在: {image_path}")
            return None
        
        # 准备评估数据
        data_info = experiment_data.get('data_info', '无数据信息')
        chart_type = experiment_data.get('chart_type', '未知图表类型')
        layout_template = experiment_data.get('layout_template', '未知布局模板')
        
        # 构建评估提示词
        evaluation_prompt = self.evaluation_prompt_template.replace(
            '{data_info}', str(data_info)
        ).replace(
            '{chart_type}', str(chart_type)
        ).replace(
            '{layout_template}', str(layout_template)
        )
        
        for attempt in range(self.config.retry_attempts):
            try:
                if self.config.evaluator_type == AIEvaluatorType.GPT4V:
                    result = self._evaluate_with_gpt4v(image_path, evaluation_prompt)
                elif self.config.evaluator_type == AIEvaluatorType.CLAUDE_VISION:
                    result = self._evaluate_with_claude_vision(image_path, evaluation_prompt)
                elif self.config.evaluator_type == AIEvaluatorType.GEMINI_VISION:
                    result = self._evaluate_with_gemini_vision(image_path, evaluation_prompt)
                else:
                    print(f"不支持的评估器类型: {self.config.evaluator_type}")
                    return None
                
                if result:
                    # 添加元数据
                    result['experiment_id'] = experiment_data.get('experiment_id', 'unknown')
                    result['image_path'] = image_path
                    result['evaluation_timestamp'] = time.time()
                    result['evaluator_type'] = self.config.evaluator_type.value
                    
                    self.evaluation_results.append(result)
                    return result
                    
            except Exception as e:
                print(f"评估图像失败 (尝试 {attempt + 1}/{self.config.retry_attempts}): {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)
        
        # 记录失败的评估
        self.failed_evaluations.append({
            "experiment_id": experiment_data.get('experiment_id', 'unknown'),
            "image_path": image_path,
            "error": "评估失败，已达到最大重试次数"
        })
        return None
    
    def _evaluate_with_gpt4v(self, image_path: str, prompt: str) -> Optional[Dict[str, Any]]:
        """使用GPT-4V进行评估"""
        if OpenAI is None:
            print("openai库未安装，无法调用GPT-4V API")
            return None
        #print(f"调用GPT-4V API")
        try:
            base64_image = self.encode_image_to_base64(image_path)
            
            # 创建OpenAI客户端
            client = OpenAI(api_key=self.config.api_key, base_url="https://aihubmix.com/v1")
            
            # 调用API
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            content = response.choices[0].message.content
            #print(f"原始内容: {content}")
            # 尝试解析JSON结果
            try:
                # 提取JSON部分 - 改进的解析逻辑
                evaluation_result = self._extract_json_from_response(content)
                if evaluation_result:
                    return evaluation_result
            except Exception as e:
                print(f"无法解析GPT-4V返回的JSON结果: {e}")
                print(f"原始内容: {content}")
                
        except Exception as e:
            print(f"GPT-4V API调用失败: {e}")
            
        return None
    
    def _evaluate_with_claude_vision(self, image_path: str, prompt: str) -> Optional[Dict[str, Any]]:
        """使用Claude Vision进行评估"""
        if anthropic is None:
            print("anthropic库未安装，无法调用Claude Vision API")
            return None
            
        try:
            base64_image = self.encode_image_to_base64(image_path)
            
            # 创建Anthropic客户端
            client = anthropic.Anthropic(api_key=self.config.api_key)
            
            # 调用API
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=self.config.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            content = response.content[0].text
            
            # 尝试解析JSON结果
            try:
                # 提取JSON部分 - 改进的解析逻辑
                evaluation_result = self._extract_json_from_response(content)
                if evaluation_result:
                    return evaluation_result
            except Exception as e:
                print(f"无法解析Claude Vision返回的JSON结果: {e}")
                print(f"原始内容: {content}")
                
        except Exception as e:
            print(f"Claude Vision API调用失败: {e}")
            
        return None
    
    def _evaluate_with_gemini_vision(self, image_path: str, prompt: str) -> Optional[Dict[str, Any]]:
        """使用Gemini Vision进行评估"""
        # 这里可以集成Google Gemini Vision API
        print("Gemini Vision评估功能待实现")
        return None
    
    def batch_evaluate_images(self, 
                            images_dir: str, 
                            experiment_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量评估图像"""
        print(f"开始批量评估图像...")
        
        images_path = Path(images_dir)
        if not images_path.exists():
            print(f"图像目录不存在: {images_dir}")
            return {"error": "图像目录不存在"}
        
        results = {
            "total_images": len(experiment_data),
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "evaluation_results": [],
            "failed_images": []
        }
        
        for i, exp_data in enumerate(experiment_data, 1):
            experiment_id = exp_data.get('experiment_id', f'exp_{i}')
            image_filename = f"{experiment_id}.png"
            image_path = images_path / image_filename
            
            print(f"正在评估图像 {i}/{len(experiment_data)}: {experiment_id}")
            
            if not image_path.exists():
                print(f"图像文件不存在: {image_path}")
                results["failed_evaluations"] += 1
                results["failed_images"].append({
                    "experiment_id": experiment_id,
                    "error": "图像文件不存在"
                })
                continue
            
            evaluation_result = self.evaluate_single_image(str(image_path), exp_data)
            
            if evaluation_result:
                results["successful_evaluations"] += 1
                results["evaluation_results"].append(evaluation_result)
                print(f"✓ 评估完成: {experiment_id}")
            else:
                results["failed_evaluations"] += 1
                results["failed_images"].append({
                    "experiment_id": experiment_id,
                    "error": "评估失败"
                })
                print(f"✗ 评估失败: {experiment_id}")
            
            # 添加延迟以避免API限制
            if i < len(experiment_data):
                time.sleep(3)
        
        # 保存评估报告
        report_path = images_path.parent / "ai_evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n批量评估完成!")
        print(f"成功: {results['successful_evaluations']}/{results['total_images']}")
        print(f"失败: {results['failed_evaluations']}/{results['total_images']}")
        print(f"评估报告已保存: {report_path}")
        
        return results
    
    def generate_evaluation_summary(self, results: Dict[str, Any]) -> str:
        """生成评估摘要报告"""
        if not results.get('evaluation_results'):
            return "无评估结果可生成摘要"
        
        evaluations = results['evaluation_results']
        
        # 计算平均分数
        total_scores = []
        data_consistency_scores = []
        layout_accuracy_scores = []
        aesthetic_quality_scores = []
        
        for eval_result in evaluations:
            total_scores.append(eval_result.get('overall_score', 0))
            data_consistency_scores.append(eval_result.get('data_consistency', {}).get('total_score', 0))
            layout_accuracy_scores.append(eval_result.get('layout_accuracy', {}).get('total_score', 0))
            aesthetic_quality_scores.append(eval_result.get('aesthetic_quality', {}).get('total_score', 0))
        
        avg_total = sum(total_scores) / len(total_scores) if total_scores else 0
        avg_data_consistency = sum(data_consistency_scores) / len(data_consistency_scores) if data_consistency_scores else 0
        avg_layout_accuracy = sum(layout_accuracy_scores) / len(layout_accuracy_scores) if layout_accuracy_scores else 0
        avg_aesthetic_quality = sum(aesthetic_quality_scores) / len(aesthetic_quality_scores) if aesthetic_quality_scores else 0
        
        # 找出最佳和最差表现
        best_eval = max(evaluations, key=lambda x: x.get('overall_score', 0))
        worst_eval = min(evaluations, key=lambda x: x.get('overall_score', 0))
        
        summary = f"""
# AI自动评估摘要报告

## 评估概览
- **总评估数量**: {len(evaluations)}
- **平均总分**: {avg_total:.2f}/30 ({avg_total/30*100:.1f}%)
- **平均数据一致性得分**: {avg_data_consistency:.2f}/10
- **平均布局准确性得分**: {avg_layout_accuracy:.2f}/10
- **平均美观度得分**: {avg_aesthetic_quality:.2f}/10

## 最佳表现
- **实验ID**: {best_eval.get('experiment_id', 'unknown')}
- **总分**: {best_eval.get('overall_score', 0):.2f}/30
- **优点**: {', '.join(best_eval.get('strengths', []))}

## 最差表现
- **实验ID**: {worst_eval.get('experiment_id', 'unknown')}
- **总分**: {worst_eval.get('overall_score', 0):.2f}/30
- **不足**: {', '.join(worst_eval.get('weaknesses', []))}

## 整体改进建议
{self._extract_common_suggestions(evaluations)}

---
*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
*评估器类型: {self.config.evaluator_type.value}*
"""
        
        return summary
    
    def _extract_common_suggestions(self, evaluations: List[Dict]) -> str:
        """提取常见改进建议"""
        all_suggestions = []
        for eval_result in evaluations:
            suggestions = eval_result.get('improvement_suggestions', [])
            all_suggestions.extend(suggestions)
        
        # 简单的建议频次统计
        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
        
        # 返回最常见的建议
        common_suggestions = sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if common_suggestions:
            return "\n".join([f"- {suggestion} (出现{count}次)" for suggestion, count in common_suggestions])
        else:
            return "无通用改进建议"
    
    def save_evaluation_results(self, output_dir: str = "evaluations"):
        """保存评估结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存详细评估结果
        results_file = output_path / "ai_evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)
        
        # 转换为CSV格式便于分析
        if self.evaluation_results:
            csv_data = []
            for result in self.evaluation_results:
                csv_row = {
                    'experiment_id': result.get('experiment_id', ''),
                    'overall_score': result.get('overall_score', 0),
                    'data_consistency_score': result.get('data_consistency', {}).get('total_score', 0),
                    'layout_accuracy_score': result.get('layout_accuracy', {}).get('total_score', 0),
                    'aesthetic_quality_score': result.get('aesthetic_quality', {}).get('total_score', 0),
                    'evaluator_type': result.get('evaluator_type', ''),
                    'evaluation_timestamp': result.get('evaluation_timestamp', 0)
                }
                csv_data.append(csv_row)
            
            csv_file = output_path / "ai_evaluation_results.csv"
            pd.DataFrame(csv_data).to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"评估结果已保存到: {output_path}")

def create_evaluator_from_config(config_file: str = "ai_evaluator_config.json") -> AIEvaluator:
    """从配置文件创建AI评估器"""
    if Path(config_file).exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        # 创建默认配置文件
        config_data = {
            "evaluator_type": "gpt4v",
            "api_key": "your-api-key-here",
            "model": "gpt-4-vision-preview",
            "max_tokens": 1500,
            "temperature": 0.1,
            "retry_attempts": 3,
            "retry_delay": 5
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        print(f"已创建默认配置文件: {config_file}")
        print("请编辑配置文件并设置您的API密钥")
    
    config = EvaluationConfig(
        evaluator_type=AIEvaluatorType(config_data["evaluator_type"]),
        api_key=config_data["api_key"],
        model=config_data.get("model", "gpt-4-vision-preview"),
        max_tokens=config_data.get("max_tokens", 1500),
        temperature=config_data.get("temperature", 0.1),
        retry_attempts=config_data.get("retry_attempts", 3),
        retry_delay=config_data.get("retry_delay", 5)
    )
    
    return AIEvaluator(config)

def main():
    """主函数"""
    print("AI自动评估器")
    print("=" * 50)
    
    # 检查生成的图像目录
    images_dir = "generated_images"
    if not Path(images_dir).exists():
        print(f"错误: 找不到图像目录 '{images_dir}'")
        print("请先运行图像生成流程")
        return
    
    try:
        # 创建AI评估器
        evaluator = create_evaluator_from_config()
        
        # 检查API密钥
        if evaluator.config.api_key == "your-api-key-here":
            print("错误: 请在ai_evaluator_config.json中设置您的API密钥")
            return
        
        # 加载实验数据
        experiment_matrix_file = "experiment_output/experimental_matrix.csv"
        if Path(experiment_matrix_file).exists():
            matrix_df = pd.read_csv(experiment_matrix_file)
            experiment_data = matrix_df.to_dict('records')
            
            # 转换为评估所需格式
            eval_data = []
            for row in experiment_data:
                eval_data.append({
                    'experiment_id': row['Run ID'],
                    'data_info': f"数据样本ID: {row['Data Sample ID']}",
                    'chart_type': row['Chart Type'],
                    'layout_template': row['Layout Template ID']
                })
        else:
            print(f"错误: 找不到实验矩阵文件 '{experiment_matrix_file}'")
            return
        
        print(f"已加载 {len(eval_data)} 个实验条件")
        
        # 批量评估图像
        results = evaluator.batch_evaluate_images(images_dir, eval_data)
        
        # 生成摘要报告
        summary = evaluator.generate_evaluation_summary(results)
        print("\n" + summary)
        
        # 保存评估结果
        evaluator.save_evaluation_results()
        
        print("\nAI自动评估完成!")
        
    except Exception as e:
        print(e)
        print(f"AI评估过程中发生错误: {e}")

if __name__ == "__main__":
    main()