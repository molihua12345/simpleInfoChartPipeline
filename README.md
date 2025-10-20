# ChartGalaxy 数据驱动信息图生成实验

基于ChartGalaxy方法论的数据驱动信息图生成质量验证实验框架。本项目实现了完整的实验流程，包括数据预处理、实验设计、提示词生成和评估框架。

## 项目概述

本实验旨在验证ChartGalaxy方法论在数据驱动信息图生成中的有效性，通过3×3×3全因子实验设计，系统评估不同数据样本、图表类型和布局模板组合的生成效果。

### 核心特性

- **数据预处理**: 自动处理MatPlotBench数据集，提取关键信息并简化数据
- **实验设计**: 基于ChartGalaxy分类系统的全因子实验矩阵
- **提示词生成**: 模块化提示词模板，支持动态变量注入
- **评估框架**: 多维度量化评估标准，包含数据一致性、布局准确性和美观度
- **自动化流程**: 一键运行完整实验流程

## 环境要求

- Python 3.8+
- 依赖包：pandas, numpy, openai, anthropic, Pillow
- AI API密钥（可选，用于自动化图像生成和评估）

## 安装和设置

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd simplePipeline
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **准备数据**
   - 确保 `benchmark_data` 目录包含完整的MatPlotBench数据集
   - 数据集应包含 `benchmark_instructions.json`、`data/` 和 `ground_truth/` 目录

4. **配置LLM服务（推荐）**
   为了获得最佳的智能化体验，建议配置LLM API：
   
   编辑 `llm_config.json`：
   ```json
   {
     "api_type": "openai",
     "api_key": "sk-xxx",
     "model": "gpt-3.5-turbo",
     "base_url":"https://aihubmix.com/v1",
     "timeout": 30,
     "max_retries": 3,
     "enable_intelligent_processing": true,
     "fallback_mode": true
   }
   ```

5. **配置AI服务（可选）**

   **配置AI图像生成**
   编辑 `ai_image_generator_config.json`：
   ```json
   {
     "generator_type": "dalle",
     "api_key": "your-actual-api-key",
     "model": "dall-e-3",
     "image_size": "1024x1024",
     "image_quality": "standard"
   }
   ```

   **配置AI评估**
   编辑 `ai_evaluator_config.json`：
   ```json
   {
     "evaluator_type": "gpt4v",
     "api_key": "your-actual-api-key",
     "model": "gpt-4-vision-preview",
     "max_tokens": 1500
   }
   ```

## 使用方法

### 快速开始

#### 基本使用
运行基础实验（规则模式）：
```bash
python run_experiment.py
```

#### 智能化使用（推荐）
运行智能化实验（LLM增强）：
```bash
python run_experiment.py --config llm_config.json
```

#### 完整AI自动化
运行端到端AI自动化实验：
```bash
python run_experiment.py --config llm_config.json --enable-ai-generation --enable-ai-evaluation
```

**注意**: 
- 如果未配置LLM API，系统将使用规则模式
- 如果未配置AI图像生成/评估API，系统将跳过对应步骤
- 系统具有自动回退机制，确保实验能够完成

### 分步执行

如需分步执行实验，可以导入相应模块：

```python
from run_experiment import ExperimentRunner

# 创建实验运行器
runner = ExperimentRunner()

# 初始化组件
runner.initialize_components()

# 分步执行
runner.step1_process_data()        # 数据预处理
runner.step2_generate_matrix()     # 生成实验矩阵
runner.step3_generate_prompts()    # 生成提示词
runner.step4_setup_evaluation()    # 设置评估框架
```

## 实验设计

### 实验矩阵 (3×3×3)

**因子1: 数据样本 (3个)**
- 从MatPlotBench数据集中选择前3个样本
- 每个样本包含数值数据和对应的绘图指令

**因子2: 图表类型 (3种)**
- Vertical Bar Chart (垂直条形图)
- Pie Chart (饼图) 
- Line Graph (折线图)

**因子3: 布局模板 (3种)**
- LT-01: Classic Centered Layout (经典居中布局)
- LT-08: Asymmetric Split Layout (非对称分割布局)
- LT-25: Immersive Overlay Layout (沉浸式叠加布局)

**总实验条件**: 27个

### AI自动化评估维度

1. **数据一致性** (Data Consistency) - 10分
   - 数值准确性、类别标签、比例关系、数据完整性

2. **布局准确性** (Layout Accuracy) - 10分
   - 元素位置、空间分配、对齐方式、层次结构

3. **美观度** (Aesthetic Quality) - 10分
   - 色彩搭配、字体排版、视觉平衡、设计一致性

### 支持的AI服务

#### LLM智能化服务
- **OpenAI GPT系列**: GPT-3.5-turbo, GPT-4等
- **Anthropic Claude系列**: Claude-3-haiku, Claude-3-sonnet等
- **自定义API**: 支持兼容OpenAI格式的其他LLM服务

#### 图像生成
- **DALL-E 3**: OpenAI的最新图像生成模型
- **Stable Diffusion**: 开源扩散模型
- **Replicate**: 多模型API平台

#### 自动评估
- **GPT-4V**: OpenAI的视觉理解模型
- **Claude Vision**: Anthropic的多模态模型

## 输出文件

实验运行后会在 `experiment_output/` 目录生成以下文件：

### 数据处理
- `processed_data/processed_samples.json`: 预处理后的数据样本

### 实验设计
- `experimental_matrix.csv`: 完整实验矩阵

### 提示词生成
- `prompts/all_prompts.json`: 所有实验条件的提示词
- `prompts/prompts.txt`: 提示词生成摘要

### AI生成结果
- `generated_images/`: AI生成的信息图图像
- `generation_report.json`: 详细生成报告

### AI评估结果
- `evaluations/ai_evaluation_results.json`: AI自动评估详细结果

### 报告
- `reports/chartgalaxy_e2e_automation_exp_comprehensive_report.md`: 实验执行报告

## 实验流程

### 第一阶段：实验准备
1. **数据预处理**: 加载MatPlotBench数据，提取关键信息，简化数据结构
2. **实验设计**: 生成3×3×3全因子实验矩阵
3. **提示词生成**: 为每个实验条件生成结构化提示词
4. **评估准备**: 创建评估模板和指南

### 第二阶段：图像生成（AI自动化）
1. 自动调用AI图像生成API（DALL-E、Stable Diffusion、Replicate等）
2. 批量处理所有实验条件的提示词，生成对应信息图
3. 自动保存生成的图像到指定目录，按实验ID命名
4. 生成详细的生成报告和统计分析

### 第三阶段：质量评估（AI自动化）
1. 使用多模态AI模型（GPT-4V、Claude Vision等）自动分析生成的图像
2. 基于预定义评估标准自动打分和生成评估报告
3. 自动填写评估模板，记录各维度得分和详细反馈
4. 计算性能指标和统计分析

### 第四阶段：结果分析（自动化）
1. 自动运行结果分析脚本
2. 生成统计报告和可视化图表
3. 自动撰写综合实验结论

## 模块说明

### data_preprocessor.py
- `MatPlotBenchProcessor`: 智能数据预处理
- LLM辅助的主题提取、数据质量评估、智能样本选择
- 支持规则模式和AI增强模式的混合处理

### experiment_matrix.py
- `ExperimentMatrix`: 生成实验设计矩阵
- 支持全因子设计和随机化

### prompt_generator.py
- `PromptGenerator`: 智能提示词生成
- LLM增强的提示词优化、上下文感知的元素建议
- 规则与AI的混合生成策略

### evaluation_framework.py
- `EvaluationFramework`: 智能评估框架
- LLM辅助的多维度评估、AI与规则评估的智能融合
- 增强的反馈和建议生成

### ai_image_generator.py
- AI图像生成模块，支持多种图像生成API
- 统一的图像生成接口
- 支持DALL-E、Stable Diffusion、Replicate等
- 批量生成和错误处理
- 生成结果统计和报告
- 自动重试机制和API限制处理

### ai_evaluator.py
- AI自动评估模块，使用多模态AI进行质量评估
- 支持GPT-4V、Claude Vision等模型
- 基于ChartGalaxy评估维度的量化评分
- 自动生成详细反馈和改进建议
- 批量评估和性能分析
- 结构化评估结果输出

### run_experiment.py
- `ExperimentRunner`: 主实验控制器
- 整合所有模块，提供完整实验流程

## 自定义配置

### 修改图表类型
在 `experiment_matrix.py` 中修改 `CHART_TYPES` 列表：

```python
CHART_TYPES = [
    "Vertical Bar Chart",
    "Horizontal Bar Chart",  # 新增
    "Pie Chart",
    "Line Graph"
]
```

### 修改布局模板
在 `experiment_matrix.py` 中修改 `LAYOUT_TEMPLATES` 列表：

```python
LAYOUT_TEMPLATES = [
    {"id": "LT-01", "name": "Classic Centered Layout"},
    {"id": "LT-02", "name": "New Layout Template"},  # 新增
    # ...
]
```

### 修改评估标准
在 `evaluation_framework.py` 中的 `_define_evaluation_criteria` 方法中修改评估标准