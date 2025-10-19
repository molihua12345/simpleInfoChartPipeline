#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI驱动图像生成模块

支持多种AI图像生成API的自动化调用，批量生成信息图。
支持DALL-E、Midjourney、Stable Diffusion等主流AI图像生成服务。

作者: lxd
日期: 2025
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import base64
from io import BytesIO

# 条件导入OpenAI库
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class ImageGeneratorType(Enum):
    """图像生成器类型枚举"""
    DALLE = "dalle"
    MIDJOURNEY = "midjourney"
    STABLE_DIFFUSION = "stable_diffusion"
    REPLICATE = "replicate"

@dataclass
class GenerationConfig:
    """图像生成配置"""
    generator_type: ImageGeneratorType
    api_key: str
    model: str = "dall-e-3"
    size: str = "1024x1024"
    quality: str = "standard"
    style: str = "natural"
    batch_size: int = 1
    retry_attempts: int = 3
    retry_delay: int = 5

class AIImageGenerator:
    """AI图像生成器类"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.generated_images = []
        self.failed_generations = []
        
    def generate_single_image(self, prompt: str, experiment_id: str, output_dir: str) -> Optional[str]:
        """生成单张图像"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        image_filename = f"{experiment_id}.png"
        image_path = output_path / image_filename
        
        for attempt in range(self.config.retry_attempts):
            try:
                if self.config.generator_type == ImageGeneratorType.DALLE:
                    success = self._generate_with_dalle(prompt, str(image_path))
                elif self.config.generator_type == ImageGeneratorType.STABLE_DIFFUSION:
                    success = self._generate_with_stable_diffusion(prompt, str(image_path))
                elif self.config.generator_type == ImageGeneratorType.REPLICATE:
                    success = self._generate_with_replicate(prompt, str(image_path))
                else:
                    print(f"不支持的生成器类型: {self.config.generator_type}")
                    return None
                
                if success:
                    self.generated_images.append({
                        "experiment_id": experiment_id,
                        "prompt": prompt,
                        "image_path": str(image_path),
                        "generator": self.config.generator_type.value,
                        "timestamp": time.time()
                    })
                    return str(image_path)
                    
            except Exception as e:
                print(f"生成图像失败 (尝试 {attempt + 1}/{self.config.retry_attempts}): {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)
        
        # 记录失败的生成
        self.failed_generations.append({
            "experiment_id": experiment_id,
            "prompt": prompt,
            "error": "生成失败，已达到最大重试次数"
        })
        return None
    
    def _generate_with_dalle(self, prompt: str, output_path: str) -> bool:
        """使用DALL-E生成图像"""
        if OpenAI is None:
            print("openai库未安装，无法调用DALL-E API")
            return False
            
        try:
            # 创建OpenAI客户端
            client = OpenAI(api_key=self.config.api_key, base_url="https://aihubmix.com/v1")
            
            # 调用图像生成API
            response = client.images.generate(
                model=self.config.model,
                prompt=prompt,
                n=1,
                size=self.config.size,
                quality=self.config.quality,
                style=self.config.style
            )
            
            # 获取图像URL
            image_url = response.data[0].url
            
            # 下载图像
            img_response = requests.get(image_url, timeout=30)
            if img_response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(img_response.content)
                return True
            else:
                print(f"图像下载失败: {img_response.status_code}")
                return False
                
        except Exception as e:
            print(f"DALL-E API调用失败: {e}")
            return False
    
    def _generate_with_stable_diffusion(self, prompt: str, output_path: str) -> bool:
        """使用Stable Diffusion生成图像"""
        # 这里可以集成Stability AI API或本地Stable Diffusion
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30
        }
        
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers=headers,
            json=data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            image_data = base64.b64decode(result['artifacts'][0]['base64'])
            
            with open(output_path, 'wb') as f:
                f.write(image_data)
            return True
        
        print(f"Stable Diffusion API错误: {response.status_code} - {response.text}")
        return False
    
    def _generate_with_replicate(self, prompt: str, output_path: str) -> bool:
        """使用Replicate API生成图像"""
        headers = {
            "Authorization": f"Token {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "version": "ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
            "input": {
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "num_outputs": 1,
                "scheduler": "K_EULER",
                "num_inference_steps": 50,
                "guidance_scale": 7.5
            }
        }
        
        # 创建预测
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 201:
            prediction = response.json()
            prediction_id = prediction['id']
            
            # 轮询结果
            max_wait_time = 300  # 5分钟
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                status_response = requests.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers=headers,
                    timeout=30
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    
                    if status_data['status'] == 'succeeded':
                        image_url = status_data['output'][0]
                        
                        # 下载图像
                        img_response = requests.get(image_url, timeout=30)
                        if img_response.status_code == 200:
                            with open(output_path, 'wb') as f:
                                f.write(img_response.content)
                            return True
                    
                    elif status_data['status'] == 'failed':
                        print(f"Replicate生成失败: {status_data.get('error', '未知错误')}")
                        return False
                
                time.sleep(5)
        
        print(f"Replicate API错误: {response.status_code} - {response.text}")
        return False
    
    def batch_generate_images(self, prompts_data: List[Dict], output_dir: str = "generated_images") -> Dict[str, Any]:
        """批量生成图像"""
        print(f"开始批量生成 {len(prompts_data)} 张图像...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {
            "total_prompts": len(prompts_data),
            "successful_generations": 0,
            "failed_generations": 0,
            "generated_images": [],
            "failed_prompts": []
        }
        
        for i, prompt_data in enumerate(prompts_data, 1):
            experiment_id = prompt_data.get('experiment_id', f'exp_{i}')
            prompt = prompt_data.get('prompt', '')
            
            print(f"正在生成图像 {i}/{len(prompts_data)}: {experiment_id}")
            
            image_path = self.generate_single_image(prompt, experiment_id, output_dir)
            
            if image_path:
                results["successful_generations"] += 1
                results["generated_images"].append({
                    "experiment_id": experiment_id,
                    "image_path": image_path,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt
                })
                print(f"✓ 成功生成: {experiment_id}")
            else:
                results["failed_generations"] += 1
                results["failed_prompts"].append({
                    "experiment_id": experiment_id,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt
                })
                print(f"✗ 生成失败: {experiment_id}")
            
            # 添加延迟以避免API限制
            if i < len(prompts_data):
                time.sleep(2)
        
        # 保存生成报告
        report_path = output_path / "generation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n批量生成完成!")
        print(f"成功: {results['successful_generations']}/{results['total_prompts']}")
        print(f"失败: {results['failed_generations']}/{results['total_prompts']}")
        print(f"生成报告已保存: {report_path}")
        
        return results
    
    def load_prompts_from_file(self, prompts_file: str) -> List[Dict]:
        """从文件加载提示词数据"""
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
        
        # 转换为标准格式
        formatted_prompts = []
        for prompt_item in prompts_data:
            formatted_prompts.append({
                "experiment_id": prompt_item.get('run_id', prompt_item.get('experiment_id', 'unknown')),
                "prompt": prompt_item.get('prompt', ''),
                "metadata": {
                    "data_sample_id": prompt_item.get('data_sample_id'),
                    "chart_type": prompt_item.get('chart_type'),
                    "layout_template_id": prompt_item.get('layout_template_id')
                }
            })
        
        return formatted_prompts

def create_generator_from_config(config_file: str = "ai_image_generator_config.json") -> AIImageGenerator:
    """从配置文件创建图像生成器"""
    if Path(config_file).exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        # 创建默认配置文件
        config_data = {
            "generator_type": "dalle",
            "api_key": "your-api-key-here",
            "model": "dall-e-3",
            "size": "1024x1024",
            "quality": "standard",
            "style": "natural",
            "batch_size": 1,
            "retry_attempts": 3,
            "retry_delay": 5
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        print(f"已创建默认配置文件: {config_file}")
        print("请编辑配置文件并设置您的API密钥")
    
    config = GenerationConfig(
        generator_type=ImageGeneratorType(config_data["generator_type"]),
        api_key=config_data["api_key"],
        model=config_data.get("model", "dall-e-3"),
        size=config_data.get("size", "1024x1024"),
        quality=config_data.get("quality", "standard"),
        style=config_data.get("style", "natural"),
        batch_size=config_data.get("batch_size", 1),
        retry_attempts=config_data.get("retry_attempts", 3),
        retry_delay=config_data.get("retry_delay", 5)
    )
    
    return AIImageGenerator(config)

def main():
    """主函数"""
    print("AI图像生成器")
    print("=" * 50)
    
    # 检查提示词文件
    prompts_file = "experiment_output/prompts/all_prompts.json"
    if not Path(prompts_file).exists():
        print(f"错误: 找不到提示词文件 '{prompts_file}'")
        print("请先运行实验生成提示词")
        return
    
    try:
        # 创建图像生成器
        generator = create_generator_from_config()
        
        # 检查API密钥
        if generator.config.api_key == "your-api-key-here":
            print("错误: 请在ai_config.json中设置您的API密钥")
            return
        
        # 加载提示词
        prompts_data = generator.load_prompts_from_file(prompts_file)
        print(f"已加载 {len(prompts_data)} 个提示词")
        
        # 批量生成图像
        results = generator.batch_generate_images(prompts_data)
        
        print("\n图像生成完成!")
        print(f"成功生成: {results['successful_generations']} 张")
        print(f"生成失败: {results['failed_generations']} 张")
        
    except Exception as e:
        print(f"图像生成过程中发生错误: {e}")

if __name__ == "__main__":
    main()