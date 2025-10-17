#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChartGalaxy数据驱动信息图生成实验主脚本

整合所有模块，实现完整的端到端自动化实验流程：
1. 数据预处理
2. 实验矩阵生成
3. 提示词生成
4. AI图像生成
5. AI自动评估
6. 实验报告生成

作者: ChartGalaxy Pipeline
日期: 2024
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# 导入自定义模块
from data_preprocessor import DataPreprocessor
from experiment_matrix import ExperimentMatrix
from prompt_generator import PromptGenerator
from evaluation_framework import EvaluationFramework
from ai_image_generator import AIImageGenerator, create_generator_from_config
from ai_evaluator import AIEvaluator, create_evaluator_from_config

class ExperimentRunner:
    """实验运行器类"""
    
    def __init__(self, 
                 benchmark_data_dir: str = "benchmark_data",
                 output_dir: str = "experiment_output",
                 config_path: str = None):
        """
        初始化实验运行器
        
        Args:
            benchmark_data_dir: MatPlotBench数据目录
            output_dir: 实验输出目录
            config_path: 配置文件路径
        """
        self.benchmark_data_dir = Path(benchmark_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 创建子目录
        self.processed_data_dir = self.output_dir / "processed_data"
        self.prompts_dir = self.output_dir / "prompts"
        self.evaluations_dir = self.output_dir / "evaluations"
        self.reports_dir = self.output_dir / "reports"
        
        for dir_path in [self.processed_data_dir, self.prompts_dir, 
                        self.evaluations_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 初始化组件
        self.data_processor = None
        self.experiment_matrix = None
        self.prompt_generator = None
        self.evaluation_framework = None
        
        # 实验状态
        self.experiment_id = None
        self.start_time = None
        self.experiment_log = []
    
    def initialize_components(self):
        """初始化所有实验组件"""
        self.log("正在初始化实验组件...")
        
        try:
            # 初始化数据处理器（传入配置）
            self.data_processor = DataPreprocessor(
                str(self.benchmark_data_dir),
                self.config
            )
            self.log("数据处理器初始化完成")
            
            # 初始化实验矩阵
            self.experiment_matrix = ExperimentMatrix()
            self.log("实验矩阵初始化完成")
            
            # 初始化提示词生成器（传入配置）
            self.prompt_generator = PromptGenerator(self.config)
            self.log("提示词生成器初始化完成")
            
            # 初始化评估框架（传入配置）
            self.evaluation_framework = EvaluationFramework(self.config)
            self.log("评估框架初始化完成")
            
            self.log("所有组件初始化完成")
            return True
            
        except Exception as e:
            self.log(f"组件初始化失败: {e}")
            return False
    
    def log(self, message: str, level: str = "INFO"):
        """记录实验日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.experiment_log.append(log_entry)
    
    def start_experiment(self, experiment_name: str = None) -> str:
        """开始实验"""
        self.start_time = datetime.now()
        # Format time string outside f-string to avoid backslash issue
        time_str = self.start_time.strftime('%Y%m%d_%H%M%S')
        self.experiment_id = experiment_name or f"exp_{time_str}"
        
        self.log(f"开始实验: {self.experiment_id}")
        self.log(f"实验开始时间: {self.start_time}")
        
        return self.experiment_id
    
    def step1_process_data(self) -> bool:
        """步骤1: 数据预处理"""
        self.log("=" * 50)
        self.log("步骤1: 数据预处理")
        self.log("=" * 50)
        
        try:
            # 生成实验数据样本
            self.log("正在生成实验数据样本...")
            sample_files = self.data_processor.generate_experiment_samples()
            self.log(f"成功生成 {len(sample_files)} 个数据样本")
            
            # 加载生成的样本文件
            processed_samples = []
            for i, sample_file in enumerate(sample_files, 1):
                self.log(f"正在加载样本文件 {i}/3: {sample_file}")
                
                try:
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        sample_data = json.load(f)
                    processed_samples.append(sample_data)
                    self.log(f"样本文件 {sample_file} 加载完成")
                    
                except Exception as e:
                    self.log(f"加载样本文件 {sample_file} 时出错: {e}", "ERROR")
                    continue
            
            # 保存处理后的数据
            processed_data_file = self.processed_data_dir / "processed_samples.json"
            with open(processed_data_file, 'w', encoding='utf-8') as f:
                json.dump(processed_samples, f, ensure_ascii=False, indent=2)
            
            self.log(f"处理后的数据已保存到: {processed_data_file}")
            self.log(f"数据预处理完成，成功处理 {len(processed_samples)} 个样本")
            
            return len(processed_samples) > 0
            
        except Exception as e:
            self.log(f"数据预处理失败: {e}", "ERROR")
            return False
    
    def step2_generate_matrix(self) -> bool:
        """步骤2: 生成实验矩阵"""
        self.log("=" * 50)
        self.log("步骤2: 生成实验矩阵")
        self.log("=" * 50)
        
        try:
            # 生成实验矩阵
            matrix_df = self.experiment_matrix.generate_matrix()
            self.log(f"生成实验矩阵，共 {len(matrix_df)} 个实验条件")
            
            # 保存实验矩阵
            matrix_file = self.output_dir / "experimental_matrix.csv"
            matrix_df.to_csv(matrix_file, index=False, encoding='utf-8-sig')
            self.log(f"实验矩阵已保存到: {matrix_file}")
            
            # 打印矩阵摘要
            self.log("实验矩阵摘要:")
            self.log(f"  - 数据样本: {matrix_df['Data Sample ID'].nunique()} 个")
            self.log(f"  - 图表类型: {matrix_df['Chart Type'].nunique()} 种")
            self.log(f"  - 布局模板: {matrix_df['Layout Template ID'].nunique()} 种")
            self.log(f"  - 总实验数: {len(matrix_df)} 个")
            
            return True
            
        except Exception as e:
            self.log(f"生成实验矩阵失败: {e}", "ERROR")
            return False
    
    def step3_generate_prompts(self) -> bool:
        """步骤3: 生成提示词"""
        self.log("=" * 50)
        self.log("步骤3: 生成提示词")
        self.log("=" * 50)
        
        # 检查提示词文件是否已存在
        prompts_file = self.prompts_dir / "all_prompts.json"
        if prompts_file.exists():
            self.log(f"提示词文件已存在，跳过生成步骤: {prompts_file}")
            return True
        
        try:
            # 加载处理后的数据
            processed_data_file = self.processed_data_dir / "processed_samples.json"
            with open(processed_data_file, 'r', encoding='utf-8') as f:
                processed_samples = json.load(f)
            
            # 加载实验矩阵
            matrix_file = self.output_dir / "experimental_matrix.csv"
            matrix_df = pd.read_csv(matrix_file)
            
            # 生成所有提示词
            self.log("正在生成提示词...")
            all_prompts = self.prompt_generator.generate_all_prompts(
                matrix_file=str(matrix_file),
                data_dir=str(self.processed_data_dir)
            )
            
            # 保存提示词
            prompts_file = self.prompts_dir / "all_prompts.json"
            text_file, json_file = self.prompt_generator.save_prompts(
                all_prompts, 
                output_file=str(self.prompts_dir / "prompts.txt"),
                json_output=str(prompts_file)
            )
            
            self.log(f"成功生成 {len(all_prompts)} 个提示词")
            self.log(f"提示词已保存到: {prompts_file}")
            
            # 生成提示词摘要
            self.prompt_generator.print_generation_summary(all_prompts)
            
            return True
            
        except Exception as e:
            self.log(f"生成提示词失败: {e}", "ERROR")
            return False
    
    def step4_setup_evaluation(self) -> bool:
        """步骤4: 设置评估框架"""
        self.log("=" * 50)
        self.log("步骤4: 设置评估框架")
        self.log("=" * 50)
        
        try:
            # 生成评估模板
            matrix_file = self.output_dir / "experimental_matrix.csv"
            batch_file = self.evaluation_framework.generate_evaluation_batch(
                str(matrix_file),
                str(self.evaluations_dir)
            )
            
            self.log(f"评估模板已生成: {batch_file}")
            self.log(f"评估指南已生成: {self.evaluations_dir / 'evaluation_guide.md'}")
            
            return True
            
        except Exception as e:
            self.log(f"设置评估框架失败: {e}", "ERROR")
            return False
    
    def step5_ai_image_generation(self) -> bool:
        """步骤5: AI图像生成"""
        self.log("=" * 50)
        self.log("步骤5: AI图像生成")
        self.log("=" * 50)
        
        try:
            # 创建AI图像生成器
            generator = create_generator_from_config("ai_image_generator_config.json")
            
            # 检查API密钥配置
            if generator.config.api_key == "your-api-key-here":
                self.log("跳过AI图像生成 - 未配置API密钥", "WARNING")
                return True
            
            # 加载提示词
            prompts_file = self.prompts_dir / "all_prompts.json"
            with open(prompts_file, 'r', encoding='utf-8') as f:
                all_prompts = json.load(f)
            
            # 准备生成数据
            generation_data = []
            for prompt_data in all_prompts:
                generation_data.append({
                    'experiment_id': prompt_data['run_id'],
                    'prompt': prompt_data['prompt'],
                    'chart_type': prompt_data.get('chart_type', ''),
                    'layout_template': prompt_data.get('layout_template_id', '')
                })
            
            # 批量生成图像
            self.log(f"开始生成 {len(generation_data)} 张图像...")
            results = generator.batch_generate_images(generation_data)
            
            self.log(f"AI图像生成完成: {results.get('successful_generations', 0)}/{results.get('total_prompts', 0)}")
            
            # 保存生成结果
            results_file = self.output_dir / "generation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            self.log(f"AI图像生成失败: {e}", "ERROR")
            return False
    
    def step6_ai_evaluation(self) -> bool:
        """步骤6: AI自动评估"""
        self.log("=" * 50)
        self.log("步骤6: AI自动评估")
        self.log("=" * 50)
        
        try:
            # 创建AI评估器
            evaluator = create_evaluator_from_config()
            
            # 检查API密钥配置
            if evaluator.config.api_key == "your-api-key-here":
                self.log("跳过AI自动评估 - 未配置API密钥", "WARNING")
                return True
            
            # 加载实验矩阵
            matrix_file = self.output_dir / "experimental_matrix.csv"
            matrix_df = pd.read_csv(matrix_file)
            
            # 准备评估数据
            eval_data = []
            for _, exp in matrix_df.iterrows():
                eval_data.append({
                    'experiment_id': exp['Run ID'],
                    'data_info': f"数据样本ID: {exp['Data Sample ID']}",
                    'chart_type': exp['Chart Type'],
                    'layout_template': exp['Layout Template ID']
                })
            
            # 批量评估图像
            images_dir = "generated_images"
            self.log(f"开始评估 {len(eval_data)} 张图像...")
            results = evaluator.batch_evaluate_images(images_dir, eval_data)
            
            self.log(f"AI自动评估完成: {results.get('successful_evaluations', 0)}/{results.get('total_images', 0)}")
            
            # 保存评估结果
            results_file = self.evaluations_dir / "ai_evaluation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            self.log(f"AI自动评估失败: {e}", "ERROR")
            return False
    
    def generate_experiment_report(self) -> str:
        """生成综合实验报告"""
        self.log("=" * 50)
        self.log("生成综合实验报告")
        self.log("=" * 50)
        
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else "未知"
        
        # 加载生成和评估结果
        generation_results = {}
        evaluation_results = {}
        
        try:
            gen_file = self.output_dir / "generation_results.json"
            if gen_file.exists():
                with open(gen_file, 'r', encoding='utf-8') as f:
                    generation_results = json.load(f)
        except:
            pass
        
        try:
            eval_file = self.evaluations_dir / "ai_evaluation_results.json"
            if eval_file.exists():
                with open(eval_file, 'r', encoding='utf-8') as f:
                    evaluation_results = json.load(f)
        except:
            pass
        
        # Format time strings outside f-string to avoid backslash issue
        start_time_str = self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else '未知'
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 生成报告内容
        report_content = self._generate_report_template(
            start_time_str, end_time_str, duration, 
            generation_results, evaluation_results
        )
        
        # 保存报告
        report_file = self.reports_dir / f"{self.experiment_id}_comprehensive_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.log(f"综合实验报告已生成: {report_file}")
        return str(report_file)
    
    def _generate_performance_metrics(self, evaluation_results: Dict[str, Any]) -> str:
        """生成性能指标文本"""
        if evaluation_results.get('status') == 'skipped' or not evaluation_results.get('evaluation_results'):
            return "- **状态**: 未执行评估或无评估数据"
        
        eval_data = evaluation_results.get('evaluation_results', [])
        if not eval_data:
            return "- **状态**: 无评估数据"
        
        # 计算平均分数
        total_scores = [result.get('overall_score', 0) for result in eval_data]
        data_scores = [result.get('data_consistency', {}).get('total_score', 0) for result in eval_data]
        layout_scores = [result.get('layout_accuracy', {}).get('total_score', 0) for result in eval_data]
        aesthetic_scores = [result.get('aesthetic_quality', {}).get('total_score', 0) for result in eval_data]
        
        avg_total = sum(total_scores) / len(total_scores) if total_scores else 0
        avg_data = sum(data_scores) / len(data_scores) if data_scores else 0
        avg_layout = sum(layout_scores) / len(layout_scores) if layout_scores else 0
        avg_aesthetic = sum(aesthetic_scores) / len(aesthetic_scores) if aesthetic_scores else 0
        
        return f"""
            - **平均总分**: {avg_total:.2f}/30 ({avg_total/30*100:.1f}%)
            - **平均数据一致性**: {avg_data:.2f}/10
            - **平均布局准确性**: {avg_layout:.2f}/10
            - **平均美观度**: {avg_aesthetic:.2f}/10
            - **最高分**: {max(total_scores) if total_scores else 0:.2f}/30
            - **最低分**: {min(total_scores) if total_scores else 0:.2f}/30
        """
    
    def _generate_next_steps(self, generation_results: Dict[str, Any], evaluation_results: Dict[str, Any]) -> str:
        """生成下一步建议"""
        steps = []
        
        if generation_results.get('status') == 'skipped':
            steps.append("1. 配置AI图像生成API密钥以启用自动图像生成")
        elif generation_results.get('failed_generations', 0) > 0:
            steps.append("1. 检查并优化失败的图像生成任务")
        
        if evaluation_results.get('status') == 'skipped':
            steps.append("2. 配置AI评估API密钥以启用自动质量评估")
        elif evaluation_results.get('failed_evaluations', 0) > 0:
            steps.append("2. 检查并重新评估失败的图像")
        
        if generation_results.get('status') != 'skipped' and evaluation_results.get('status') != 'skipped':
            steps.extend([
                "3. 分析评估结果，识别最佳实践模式",
                "4. 基于评估反馈优化提示词模板",
                "5. 进行A/B测试验证改进效果",
                "6. 扩展实验到更多数据样本和图表类型"
            ])
        else:
            steps.extend([
                "3. 完成API配置后重新运行完整实验",
                "4. 分析端到端自动化流程的性能表现"
            ])
        
        return "\n".join(steps)
    
    def run_full_experiment(self, experiment_name: str = None) -> bool:
        """运行完整的端到端自动化实验流程"""
        try:
            # 开始实验
            exp_id = self.start_experiment(experiment_name)
            
            # 初始化组件
            if not self.initialize_components():
                self.log("组件初始化失败，实验终止", "ERROR")
                return False
            
            # 执行实验步骤
            steps = [
                ("数据预处理", self.step1_process_data),
                ("生成实验矩阵", self.step2_generate_matrix),
                ("生成提示词", self.step3_generate_prompts),
                ("设置评估框架", self.step4_setup_evaluation),
                ("AI图像生成", self.step5_ai_image_generation),
                ("AI自动评估", self.step6_ai_evaluation)
            ]
            
            for step_name, step_func in steps:
                self.log(f"开始执行: {step_name}")
                if not step_func():
                    self.log(f"{step_name} 执行失败，实验终止", "ERROR")
                    return False
                self.log(f"{step_name} 执行完成")
            
            # 生成综合实验报告
            report_file = self.generate_experiment_report()
            
            self.log("=" * 60)
            self.log("🎉 端到端AI自动化实验执行完成！")
            self.log("=" * 60)
            self.log(f"实验ID: {exp_id}")
            self.log(f"综合报告: {report_file}")
            self.log(f"输出目录: {self.output_dir}")
            
            return True
            
        except Exception as e:
            self.log(f"实验执行过程中发生错误: {e}", "ERROR")
            return False
    
    def analyze_results(self, evaluations_dir: str = None) -> Optional[str]:
        """分析评估结果"""
        if not evaluations_dir:
            evaluations_dir = str(self.evaluations_dir)
        
        try:
            self.log("开始分析评估结果...")
            
            # 分析评估结果
            results_df = self.evaluation_framework.analyze_evaluation_results(evaluations_dir)
            
            if results_df.empty:
                self.log("未找到评估结果文件", "WARNING")
                return None
            
            # 生成分析报告
            report_file = self.reports_dir / "evaluation_analysis_report.md"
            self.evaluation_framework.generate_evaluation_report(
                results_df, 
                str(report_file)
            )
            
            self.log(f"评估分析报告已生成: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.log(f"分析评估结果时发生错误: {e}", "ERROR")
            return None
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            'llm_config': {},
            'random_seed': 42,
            'data_quality_threshold': 0.5,
            'enable_intelligent_processing': True,
            'fallback_mode': True
        }
        
        if config_path is None:
            # 尝试加载默认配置文件
            config_files = ['llm_config.json', 'config.json']
            for config_file in config_files:
                config_path = Path(self.benchmark_data_dir).parent / config_file
                if config_path.exists():
                    break
            else:
                print("未找到配置文件，使用默认配置")
                return default_config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 合并默认配置
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
            return default_config

    def _generate_report_template(self, start_time_str: str, end_time_str: str, 
                                duration, generation_results: Dict[str, Any], 
                                evaluation_results: Dict[str, Any]) -> str:
        return f"""
            # ChartGalaxy端到端AI自动化信息图生成实验报告

            ## 🎯 实验基本信息

            - **实验ID**: {self.experiment_id}
            - **实验类型**: 端到端AI自动化数据驱动信息图生成
            - **开始时间**: {start_time_str}
            - **结束时间**: {end_time_str}
            - **实验耗时**: {str(duration)}
            - **实验目标**: 验证ChartGalaxy方法论的端到端自动化能力

            ## 🔄 自动化流程状态

            - **数据预处理**: ✅ 自动化完成
            - **实验设计**: ✅ 自动化完成
            - **提示词生成**: ✅ 自动化完成
            - **图像生成**: {'✅ AI自动化' if generation_results.get('status') != 'skipped' else '⏭️ 已跳过'}
            - **质量评估**: {'✅ AI自动化' if evaluation_results.get('status') != 'skipped' else '⏭️ 已跳过'}

            ## 📊 实验设计

            ### 数据来源
            - **数据集**: MatPlotBench
            - **数据目录**: {self.benchmark_data_dir}
            - **选择样本**: 前3个数据样本

            ## 🎨 AI图像生成结果

            - **状态**: {generation_results.get('status', '未执行')}
            - **总尝试数**: {generation_results.get('total_prompts', 0)}
            - **成功生成**: {generation_results.get('successful_generations', 0)}
            - **失败生成**: {generation_results.get('failed_generations', 0)}
            - **成功率**: {generation_results.get('successful_generations', 0) / max(generation_results.get('total_prompts', 1), 1) * 100:.1f}%

            ## 🔍 AI自动评估结果

            - **状态**: {evaluation_results.get('status', '未执行')}
            - **总评估数**: {evaluation_results.get('total_images', 0)}
            - **成功评估**: {evaluation_results.get('successful_evaluations', 0)}
            - **失败评估**: {evaluation_results.get('failed_evaluations', 0)}
            - **成功率**: {evaluation_results.get('successful_evaluations', 0) / max(evaluation_results.get('total_images', 1), 1) * 100:.1f}%

            ## 📈 性能指标

            {self._generate_performance_metrics(evaluation_results)}

            ## 实验输出文件

            ### 数据处理
            - `{self.processed_data_dir / 'processed_samples.json'}`: 预处理后的数据样本

            ### 实验设计
            - `{self.output_dir / 'experimental_matrix.csv'}`: 完整实验矩阵

            ### 提示词生成
            - `{self.prompts_dir / 'all_prompts.json'}`: 所有实验条件的提示词
            - `{self.prompts_dir / 'generation_summary.txt'}`: 提示词生成摘要

            ### AI生成结果
            - `generated_images/`: AI生成的信息图图像
            - `{self.output_dir / 'generation_results.json'}`: 图像生成结果统计

            ### AI评估结果
            - `{self.evaluations_dir / 'ai_evaluation_results.json'}`: AI自动评估结果
            - `{self.evaluations_dir / 'evaluation_guide.md'}`: 评估指南文档

            ## 🚀 下一步操作

            {self._generate_next_steps(generation_results, evaluation_results)}

            ## 实验日志

            {chr(96)*3}
            {"###########next#############".join(self.experiment_log)}
            {chr(96)*3}

            ---
            *报告生成时间: {end_time_str}*  """

def main():
    """主函数"""
    print("ChartGalaxy 端到端AI自动化信息图生成实验")
    print("=" * 60)
    
    # 检查数据目录
    benchmark_dir = "benchmark_data"
    if not Path(benchmark_dir).exists():
        print(f"错误: 找不到数据目录 '{benchmark_dir}'")
        print("请确保MatPlotBench数据已正确放置在benchmark_data目录中")
        return
    
    # 创建实验运行器（加载配置）
    runner = ExperimentRunner(
        benchmark_data_dir=benchmark_dir,
        output_dir="experiment_output",
        config_path="llm_config.json"
    )
    
    # 运行完整的端到端自动化实验
    success = runner.run_full_experiment("chartgalaxy_e2e_automation_exp")
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 端到端AI自动化实验执行成功完成！")
        print("=" * 60)
        print("\n✅ 已完成的自动化流程:")
        print("1. ✅ 数据预处理")
        print("2. ✅ 实验矩阵生成")
        print("3. ✅ 提示词生成")
        print("4. 🎨 AI图像生成 (如已配置API)")
        print("5. 🔍 AI自动评估 (如已配置API)")
        print("6. ✅ 综合报告生成")
        print(f"\n📁 所有输出文件位于: {runner.output_dir}")
        print(f"📋 查看综合报告了解详细结果")
    else:
        print("\n" + "=" * 60)
        print("❌ 实验执行失败！")
        print("=" * 60)
        print("请查看日志信息排查问题")

if __name__ == "__main__":
    main()