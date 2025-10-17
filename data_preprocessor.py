#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据驱动信息图生成实验 - 数据预处理模块

基于MatPlotBench数据集，实现从原始数据到结构化JSON的转换流程。
按照ChartGalaxy方法论进行数据抽样、主题提取和补充元素生成。

作者: ChartGalaxy Pipeline
日期: 2024
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
        """加载指定样本的数据和查询指令"""
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
        elif data_file.suffix == '.json':
            with open(data_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            # 简化JSON数据处理，提取主要数值数据
            data = self._extract_data_from_json(json_data)
        else:
            data = pd.read_csv(data_file, sep='\t')
        
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
    
    def simplify_data(self, data: pd.DataFrame, max_points: int = 6) -> List[Dict[str, Any]]:
        """简化数据，确保数据点在3-6个之间"""
        # 如果数据太多，进行抽样或聚合
        if len(data) > max_points:
            # 尝试找到数值列进行聚合
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 1:
                # 按数值大小排序，取前max_points个
                data = data.nlargest(max_points, numeric_cols[0])
            else:
                # 随机抽样
                data = data.sample(n=max_points, random_state=42)
        
        # 转换为标准格式
        result = []
        
        # 尝试识别类别列和数值列
        if len(data.columns) >= 2:
            cat_col = data.columns[0]
            val_col = data.columns[1]
            
            for _, row in data.iterrows():
                try:
                    # 尝试转换为数值，如果失败则使用0
                    value = float(row[val_col]) if pd.notna(row[val_col]) else 0
                except (ValueError, TypeError):
                    # 对于无法转换的字符串（如'Chrome 74.0'），使用0作为默认值
                    value = 0
                
                result.append({
                    'category': str(row[cat_col]),
                    'value': value
                })
        else:
            # 单列数据处理
            for i, (_, row) in enumerate(data.iterrows()):
                try:
                    # 尝试转换为数值，如果失败则使用0
                    value = float(row.iloc[0]) if pd.notna(row.iloc[0]) else 0
                except (ValueError, TypeError):
                    # 对于无法转换的字符串，使用0作为默认值
                    value = 0
                
                result.append({
                    'category': f'Item {i+1}',
                    'value': value
                })
        
        return result[:max_points]
    
    def extract_topic(self, instruction: str, data: List[Dict]) -> str:
        """使用LLM API从指令中智能提取主题"""
        try:
            # 尝试调用LLM API进行智能主题提取
            topic = self._extract_topic_with_llm(instruction, data)
            if topic:
                return topic
        except Exception as e:
            print(f"LLM主题提取失败，使用备用方法: {e}")
        
        # 备用方法：基于关键词的智能主题提取
        return self._extract_topic_fallback(instruction, data)
    
    def _extract_topic_with_llm(self, instruction: str, data: List[Dict]) -> Optional[str]:
        """使用LLM API进行智能主题提取"""
        if not self.llm_config.get('api_key'):
            return None
            
        # 构建数据概览
        data_summary = self._generate_data_summary(data)
        
        prompt = f"""
        请根据以下指令和数据概览，提取一个简洁的主题词（不超过10个字符）：
        
        指令：{instruction}
        
        数据概览：{data_summary}
        
        请只返回主题词，不要其他解释。
        """
        
        try:
            response = self._call_llm_api(prompt)
            if response and len(response.strip()) <= 10:
                return response.strip()
        except Exception as e:
            print(f"LLM API调用失败: {e}")
        
        return None
    
    def _extract_topic_fallback(self, instruction: str, data: List[Dict]) -> str:
        """备用主题提取方法"""
        keywords = {
            '销售分析': ['sales', 'revenue', '销售', '营收', '收入', '业绩', '利润'],
            '用户分析': ['user', 'customer', '用户', '客户', '人群', '群体', '会员'],
            '产品分析': ['product', 'item', '产品', '商品', '服务', '项目', '品类'],
            '时间趋势': ['time', 'date', 'month', 'year', '时间', '月份', '年度', '季度', '趋势'],
            '地区分布': ['region', 'location', '地区', '城市', '省份', '区域', '分布'],
            '市场分析': ['market', 'competition', '市场', '竞争', '份额', '占比'],
            '财务分析': ['finance', 'cost', 'budget', '财务', '成本', '预算', '投资']
        }
        
        # 智能匹配最相关的主题
        instruction_lower = instruction.lower()
        topic_scores = {}
        for topic, words in keywords.items():
            score = sum(1 for word in words if word in instruction_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        
        # 基于数据列名推断主题
        if data:
            columns = list(data[0].keys()) if data else []
            for topic, words in keywords.items():
                if any(any(word in col.lower() for word in words) for col in columns):
                    return topic
        
        # 默认主题
        categories = [item['category'] for item in data]
        return f"{', '.join(categories[:3])}数据分析"
    
    def analyze_data_characteristics(self, data: List[Dict]) -> Dict[str, Any]:
        """智能分析数据特征"""
        if not data:
            return {}
        
        characteristics = {
            'row_count': len(data),
            'column_count': len(data[0]) if data else 0,
            'columns': list(data[0].keys()) if data else [],
            'data_types': {},
            'missing_values': {},
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'statistics': {},
            'data_quality_score': 0.0,
            'recommended_charts': []
        }
        
        if not data:
            return characteristics
        
        # 智能分析每列的数据类型和统计信息
        for column in characteristics['columns']:
            column_analysis = self._analyze_column(data, column)
            characteristics['data_types'][column] = column_analysis['type']
            characteristics['missing_values'][column] = column_analysis['missing_count']
            characteristics['statistics'][column] = column_analysis['statistics']
            
            # 分类到不同类型
            if column_analysis['type'] == 'numeric':
                characteristics['numeric_columns'].append(column)
            elif column_analysis['type'] == 'datetime':
                characteristics['datetime_columns'].append(column)
            else:
                characteristics['categorical_columns'].append(column)
        
        # 计算数据质量分数
        characteristics['data_quality_score'] = self._calculate_data_quality_score(characteristics)
        
        # 智能推荐图表类型
        characteristics['recommended_charts'] = self._recommend_chart_types(characteristics)
        
        return characteristics
    
    def _analyze_column(self, data: List[Dict], column: str) -> Dict[str, Any]:
        """分析单个列的特征"""
        values = [item.get(column) for item in data]
        non_null_values = [v for v in values if v is not None and str(v).strip() != '']
        
        analysis = {
            'missing_count': len(data) - len(non_null_values),
            'type': 'categorical',
            'statistics': {}
        }
        
        if not non_null_values:
            return analysis
        
        # 检测数据类型
        numeric_count = 0
        datetime_count = 0
        
        for value in non_null_values[:20]:  # 检查前20个值
            # 检测数值类型
            try:
                float(str(value))
                numeric_count += 1
            except ValueError:
                pass
            
            # 检测日期时间类型
            if self._is_datetime_like(str(value)):
                datetime_count += 1
        
        sample_size = min(20, len(non_null_values))
        
        if datetime_count / sample_size > 0.6:
            analysis['type'] = 'datetime'
        elif numeric_count / sample_size > 0.8:
            analysis['type'] = 'numeric'
            # 计算数值统计
            numeric_values = []
            for v in non_null_values:
                try:
                    numeric_values.append(float(v))
                except ValueError:
                    continue
            
            if numeric_values:
                analysis['statistics'] = {
                    'min': min(numeric_values),
                    'max': max(numeric_values),
                    'mean': sum(numeric_values) / len(numeric_values),
                    'count': len(numeric_values)
                }
        else:
            analysis['type'] = 'categorical'
            # 计算分类统计
            unique_values = list(set(str(v) for v in non_null_values))
            analysis['statistics'] = {
                'unique_count': len(unique_values),
                'most_common': unique_values[:5] if len(unique_values) <= 5 else unique_values[:5]
            }
        
        return analysis
    
    def _is_datetime_like(self, value: str) -> bool:
        """检测是否为日期时间格式"""
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{4}年\d{1,2}月',   # 中文日期
            r'Q[1-4]',             # 季度
            r'\d{4}Q[1-4]'         # 年份+季度
        ]
        
        import re
        for pattern in datetime_patterns:
            if re.search(pattern, value):
                return True
        return False
    
    def _calculate_data_quality_score(self, characteristics: Dict) -> float:
        """计算数据质量分数"""
        if not characteristics['columns']:
            return 0.0
        
        score = 1.0
        
        # 缺失值惩罚
        total_missing = sum(characteristics['missing_values'].values())
        total_cells = characteristics['row_count'] * characteristics['column_count']
        if total_cells > 0:
            missing_ratio = total_missing / total_cells
            score *= (1 - missing_ratio * 0.5)
        
        # 数据类型多样性奖励
        type_diversity = len(set(characteristics['data_types'].values()))
        score *= (1 + type_diversity * 0.1)
        
        # 数据量适中性
        row_count = characteristics['row_count']
        if 3 <= row_count <= 20:
            score *= 1.2
        elif row_count < 3:
            score *= 0.8
        
        return min(1.0, score)
    
    def _recommend_chart_types(self, characteristics: Dict) -> List[str]:
        """基于数据特征推荐图表类型"""
        recommendations = []
        
        numeric_cols = len(characteristics['numeric_columns'])
        categorical_cols = len(characteristics['categorical_columns'])
        datetime_cols = len(characteristics['datetime_columns'])
        row_count = characteristics['row_count']
        
        # 基于数据结构推荐
        if datetime_cols > 0 and numeric_cols > 0:
            recommendations.extend(['line', 'area'])
        
        if categorical_cols == 1 and numeric_cols == 1:
            if row_count <= 8:
                recommendations.extend(['pie', 'bar'])
            else:
                recommendations.append('bar')
        
        if numeric_cols >= 2:
            recommendations.append('scatter')
        
        if numeric_cols >= 1:
            recommendations.append('histogram')
        
        # 去重并限制数量
        return list(dict.fromkeys(recommendations))[:3]
    
    def determine_chart_type(self, data: List[Dict], instruction: str) -> str:
        """智能确定最适合的图表类型"""
        characteristics = self.analyze_data_characteristics(data)
        
        # 使用LLM进行智能图表类型推荐
        llm_recommendation = self._get_llm_chart_recommendation(instruction, characteristics)
        if llm_recommendation:
            return llm_recommendation
        
        # 备用方案：基于规则的智能推荐
        return self._determine_chart_type_by_rules(characteristics, instruction)
    
    def _get_llm_chart_recommendation(self, instruction: str, characteristics: Dict) -> Optional[str]:
        """使用LLM推荐图表类型"""
        if not self.llm_config.get('api_key'):
            return None
        
        prompt = f"""
        根据以下数据特征和用户指令，推荐最适合的图表类型：
        
        数据特征：
        - 数据量：{characteristics['row_count']}行
        - 数值字段：{characteristics['numeric_columns']}
        - 分类字段：{characteristics['categorical_columns']}
        - 时间字段：{characteristics['datetime_columns']}
        
        用户指令：{instruction}
        
        请从以下选项中选择一个最适合的图表类型：
        bar, line, pie, scatter, histogram, heatmap, area
        
        只返回图表类型名称，不要其他解释。
        """
        
        try:
            response = self._call_llm_api(prompt)
            if response and response.strip().lower() in ['bar', 'line', 'pie', 'scatter', 'histogram', 'heatmap', 'area']:
                return response.strip().lower()
        except Exception as e:
            print(f"LLM图表推荐失败: {e}")
        
        return None
    
    def _determine_chart_type_by_rules(self, characteristics: Dict, instruction: str) -> str:
        """基于规则的智能图表类型确定"""
        # 增强的关键词映射
        chart_keywords = {
            'bar': ['比较', 'compare', '对比', 'vs', '排名', 'ranking', '柱状', '条形'],
            'line': ['趋势', 'trend', '时间', 'time', '变化', 'change', '发展', '走势', '线性'],
            'pie': ['占比', 'proportion', '分布', 'distribution', '百分比', 'percentage', '饼图', '构成'],
            'scatter': ['关系', 'relationship', '相关', 'correlation', '散点', '关联'],
            'histogram': ['分布', 'distribution', '频率', 'frequency', '直方图', '统计'],
            'heatmap': ['热力', 'heatmap', '密度', 'density', '矩阵'],
            'area': ['面积', 'area', '堆叠', 'stack', '累积']
        }
        
        # 智能关键词匹配
        instruction_lower = instruction.lower()
        chart_scores = {}
        for chart_type, keywords in chart_keywords.items():
            score = sum(2 if keyword in instruction_lower else 0 for keyword in keywords)
            if score > 0:
                chart_scores[chart_type] = score
        
        # 基于数据特征的智能推荐
        numeric_cols = characteristics['numeric_columns']
        categorical_cols = characteristics['categorical_columns']
        datetime_cols = characteristics['datetime_columns']
        row_count = characteristics['row_count']
        
        # 数据特征评分
        if not chart_scores:
            if datetime_cols and numeric_cols:
                chart_scores['line'] = 3
            elif len(numeric_cols) >= 2 and row_count <= 100:
                chart_scores['scatter'] = 3
            elif len(categorical_cols) == 1 and len(numeric_cols) == 1:
                if row_count <= 10:
                    chart_scores['pie'] = 2
                else:
                    chart_scores['bar'] = 3
            elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                chart_scores['bar'] = 2
            elif len(numeric_cols) >= 1:
                chart_scores['histogram'] = 2
        
        # 返回得分最高的图表类型
        if chart_scores:
            return max(chart_scores, key=chart_scores.get)
        
        return 'bar'  # 默认选择
    
    def generate_supplementary_elements(self, topic: str, data: List[Dict]) -> Dict[str, Any]:
        """智能生成补充元素（文本描述和图标关键词）"""
        try:
            # 分析数据特征
            characteristics = self.analyze_data_characteristics(data)
            
            # 使用LLM生成智能描述
            llm_text = self._generate_llm_description(topic, data, characteristics)
            
            # 生成基础描述
            fallback_text = self._generate_fallback_description(data, characteristics)
            
            # 选择最佳描述
            text = llm_text if llm_text else fallback_text
            
            # 智能生成图标关键词
            icon_keywords = self._generate_smart_icon_keywords(topic, characteristics)
            
            return {
                "text": text,
                "icon_keywords": icon_keywords[:4],  # 限制为4个关键词
                "data_insights": self._extract_data_insights(data, characteristics)
            }
        except Exception as e:
            print(f"生成补充元素时出错: {e}")
            return self._generate_default_elements(topic, data)
    
    def _generate_llm_description(self, topic: str, data: List[Dict], characteristics: Dict) -> Optional[str]:
        """使用LLM生成智能描述"""
        if not self.llm_config.get('api_key'):
            return None
        
        data_summary = self._create_data_summary_for_llm(data, characteristics)
        
        prompt = f"""
        基于以下数据分析结果，生成一段简洁的数据洞察描述（不超过50字）：
        
        主题：{topic}
        数据概况：{data_summary}
        
        请生成一段专业、简洁的数据分析描述，突出关键发现。
        """
        
        try:
            response = self._call_llm_api(prompt)
            if response and len(response.strip()) <= 100:
                return response.strip()
        except Exception as e:
            print(f"LLM描述生成失败: {e}")
        
        return None
    
    def _generate_fallback_description(self, data: List[Dict], characteristics: Dict) -> str:
        """生成备用描述"""
        if not data:
            return "数据分析结果显示相关趋势和模式。"
        
        try:
            values = [item.get('value', 0) for item in data if 'value' in item]
            if values:
                max_item = max(data, key=lambda x: x.get('value', 0))
                avg_value = sum(values) / len(values)
                
                return f"数据显示{max_item.get('category', '某项')}表现最佳（{max_item.get('value', 0):.1f}），" \
                       f"平均值为{avg_value:.1f}。"
            else:
                return f"分析了{len(data)}项数据，发现了重要的业务洞察。"
        except Exception:
            return "数据分析揭示了有价值的业务趋势。"
    
    def _generate_smart_icon_keywords(self, topic: str, characteristics: Dict) -> List[str]:
        """智能生成图标关键词"""
        base_keywords = ["chart", "data", "analysis"]
        
        # 基于主题的关键词
        topic_keywords = {
            '销售': ['sales', 'money', 'growth', 'revenue'],
            '收入': ['finance', 'profit', 'revenue', 'income'],
            '利润': ['profit', 'finance', 'money', 'growth'],
            '市场': ['market', 'trend', 'competition', 'share'],
            '用户': ['user', 'customer', 'people', 'audience'],
            '产品': ['product', 'item', 'service', 'offering'],
            '时间': ['time', 'calendar', 'clock', 'schedule'],
            '地区': ['location', 'map', 'region', 'geography']
        }
        
        # 匹配主题关键词
        for key, keywords in topic_keywords.items():
            if key in topic:
                base_keywords.extend(keywords[:2])
                break
        
        # 基于数据特征的关键词
        if characteristics.get('datetime_columns'):
            base_keywords.append('timeline')
        if len(characteristics.get('numeric_columns', [])) >= 2:
            base_keywords.append('correlation')
        if characteristics.get('row_count', 0) > 10:
            base_keywords.append('big-data')
        
        # 去重并返回
        return list(dict.fromkeys(base_keywords))
    
    def _extract_data_insights(self, data: List[Dict], characteristics: Dict) -> Dict[str, Any]:
        """提取数据洞察"""
        insights = {
            'data_quality': characteristics.get('data_quality_score', 0),
            'recommended_charts': characteristics.get('recommended_charts', []),
            'key_findings': []
        }
        
        # 提取关键发现
        if data and 'value' in data[0]:
            values = [item.get('value', 0) for item in data]
            if values:
                max_val = max(values)
                min_val = min(values)
                if max_val > min_val * 2:
                    insights['key_findings'].append('数据存在显著差异')
                if len(set(values)) == len(values):
                    insights['key_findings'].append('所有数值均不相同')
        
        return insights
    
    def _generate_default_elements(self, topic: str, data: List[Dict]) -> Dict[str, Any]:
        """生成默认补充元素"""
        return {
            "text": f"基于{topic}的数据分析显示了重要的业务洞察。",
            "icon_keywords": ["chart", "data", "analysis", "business"],
            "data_insights": {
                'data_quality': 0.8,
                'recommended_charts': ['bar'],
                'key_findings': ['数据分析完成']
            }
        }
    
    def _create_data_summary_for_llm(self, data: List[Dict], characteristics: Dict) -> str:
        """为LLM创建数据摘要"""
        summary_parts = []
        
        # 基本信息
        summary_parts.append(f"数据量: {len(data)}条")
        
        # 字段信息
        if characteristics.get('columns'):
            summary_parts.append(f"字段: {', '.join(characteristics['columns'][:3])}")
        
        # 数值统计
        if data and 'value' in data[0]:
            values = [item.get('value', 0) for item in data]
            if values:
                summary_parts.append(f"数值范围: {min(values):.1f} - {max(values):.1f}")
        
        return "; ".join(summary_parts)
    
    def process_sample(self, sample_id: str) -> Dict[str, Any]:
        """智能处理单个样本，生成完整的JSON结构"""
        try:
            # 加载原始数据
            raw_data, instruction = self.load_sample_data(sample_id)
            
            # 智能简化数据
            simplified_data = self.simplify_data(raw_data)
            
            # 数据质量检查
            quality_check = self._perform_data_quality_check(simplified_data)
            if not quality_check['is_valid']:
                print(f"数据质量警告 {sample_id}: {quality_check['issues']}")
                simplified_data = self._fix_data_quality_issues(simplified_data, quality_check)
            
            # 智能提取主题
            topic = self.extract_topic(instruction, simplified_data)
            
            # 确定最佳图表类型
            chart_type = self.determine_chart_type(simplified_data, instruction)
            
            # 生成智能补充元素
            elements = self.generate_supplementary_elements(topic, simplified_data)
            
            # 构建增强的JSON结构
            result = {
                "sample_id": f"MPB_case_{sample_id}_v2",
                "topic": topic,
                "query": instruction,
                "data": simplified_data,
                "chart_type": chart_type,
                "elements": elements,
                "metadata": {
                    "processing_timestamp": datetime.now().isoformat(),
                    "data_quality_score": quality_check.get('quality_score', 0.8),
                    "processing_method": "intelligent_pipeline"
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            # 返回增强的默认样本
            return self._create_enhanced_default_sample(sample_id, str(e))
    
    def _perform_data_quality_check(self, data: List[Dict]) -> Dict[str, Any]:
        """执行数据质量检查"""
        check_result = {
            'is_valid': True,
            'issues': [],
            'quality_score': 1.0
        }
        
        if not data:
            check_result['is_valid'] = False
            check_result['issues'].append('数据为空')
            check_result['quality_score'] = 0.0
            return check_result
        
        # 检查数据结构一致性
        if len(data) > 1:
            first_keys = set(data[0].keys())
            for i, item in enumerate(data[1:], 1):
                if set(item.keys()) != first_keys:
                    check_result['issues'].append(f'第{i+1}行数据结构不一致')
                    check_result['quality_score'] *= 0.9
        
        # 检查缺失值
        missing_count = 0
        total_values = 0
        for item in data:
            for key, value in item.items():
                total_values += 1
                if value is None or str(value).strip() == '':
                    missing_count += 1
        
        if missing_count > 0:
            missing_ratio = missing_count / total_values
            check_result['issues'].append(f'存在{missing_count}个缺失值')
            check_result['quality_score'] *= (1 - missing_ratio * 0.5)
        
        # 检查数据量
        if len(data) < 3:
            check_result['issues'].append('数据量过少')
            check_result['quality_score'] *= 0.8
        elif len(data) > 20:
            check_result['issues'].append('数据量过多，建议抽样')
            check_result['quality_score'] *= 0.9
        
        # 检查数值字段
        numeric_fields = []
        for key in data[0].keys():
            numeric_count = 0
            for item in data:
                try:
                    float(str(item[key]))
                    numeric_count += 1
                except ValueError:
                    pass
            if numeric_count / len(data) > 0.8:
                numeric_fields.append(key)
        
        if not numeric_fields:
            check_result['issues'].append('缺少数值字段')
            check_result['quality_score'] *= 0.7
        
        # 设置最终验证状态
        if check_result['quality_score'] < 0.5:
            check_result['is_valid'] = False
        
        return check_result
    
    def _fix_data_quality_issues(self, data: List[Dict], quality_check: Dict) -> List[Dict]:
        """修复数据质量问题"""
        if not data:
            return self._generate_sample_data()
        
        fixed_data = []
        
        for item in data:
            fixed_item = {}
            for key, value in item.items():
                # 修复缺失值
                if value is None or str(value).strip() == '':
                    if key.lower() in ['value', 'amount', 'count', 'number']:
                        fixed_item[key] = 0
                    elif key.lower() in ['category', 'name', 'label']:
                        fixed_item[key] = f'未知{len(fixed_data)+1}'
                    else:
                        fixed_item[key] = '未知'
                else:
                    fixed_item[key] = value
            fixed_data.append(fixed_item)
        
        # 确保至少有3条数据
        while len(fixed_data) < 3:
            fixed_data.append({
                'category': f'项目{len(fixed_data)+1}',
                'value': random.randint(10, 100)
            })
        
        return fixed_data
    
    def _generate_sample_data(self) -> List[Dict]:
        """生成示例数据"""
        categories = ['产品A', '产品B', '产品C', '产品D', '产品E']
        return [
            {'category': cat, 'value': random.randint(50, 200)}
            for cat in categories[:random.randint(3, 5)]
        ]
    
    def _create_enhanced_default_sample(self, sample_id: str, error_msg: str) -> Dict[str, Any]:
        """创建增强的默认样本数据"""
        return {
            "sample_id": f"MPB_case_{sample_id}_v2_default",
            "topic": "示例数据分析",
            "query": "创建一个数据可视化图表",
            "data": [
                {"category": "Q1", "value": 120},
                {"category": "Q2", "value": 150},
                {"category": "Q3", "value": 135},
                {"category": "Q4", "value": 180}
            ],
            "chart_type": "bar",
            "elements": {
                "text": "数据显示Q4表现最佳，实现了显著增长。",
                "icon_keywords": ["chart", "data", "growth", "business"],
                "data_insights": {
                    'data_quality': 0.8,
                    'recommended_charts': ['bar', 'line'],
                    'key_findings': ['Q4表现突出', '整体呈上升趋势']
                }
            },
            "metadata": {
                "processing_timestamp": datetime.now().isoformat(),
                "data_quality_score": 0.8,
                "processing_method": "fallback_default",
                "error_info": error_msg
            }
        }
    
    def _create_default_sample(self, sample_id: str) -> Dict[str, Any]:
        """创建默认样本数据"""
        return {
            "sample_id": f"MPB_case_{sample_id}_v1",
            "topic": "示例数据分析",
            "query": "创建一个数据可视化图表",
            "data": [
                {"category": "Q1", "value": 120},
                {"category": "Q2", "value": 150},
                {"category": "Q3", "value": 135},
                {"category": "Q4", "value": 180}
            ],
            "elements": {
                "text": "数据显示Q4表现最佳，实现了显著增长。",
                "icon_keywords": ["chart", "data", "growth", "business"]
            }
        }
    
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
    
    def _generate_data_summary(self, data: List[Dict]) -> str:
        """生成数据概览"""
        if not data:
            return "无数据"
        
        summary_parts = []
        
        # 数据量
        summary_parts.append(f"数据量: {len(data)}条")
        
        # 字段信息
        if data:
            columns = list(data[0].keys())
            summary_parts.append(f"字段: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
        
        # 数值字段统计
        numeric_fields = []
        for key in data[0].keys() if data else []:
            try:
                values = [float(item[key]) for item in data[:10] if str(item[key]).replace('.', '').replace('-', '').isdigit()]
                if values:
                    numeric_fields.append(key)
            except:
                continue
        
        if numeric_fields:
            summary_parts.append(f"数值字段: {', '.join(numeric_fields[:3])}")
        
        return "; ".join(summary_parts)
    
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
                max_tokens=50,
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
            client = anthropic.Anthropic(
                api_key=api_key,
                timeout=self.llm_config.get('timeout', 30)
            )
            
            response = client.messages.create(
                model=self.llm_config.get('model', 'claude-3-haiku-20240307'),
                max_tokens=50,
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