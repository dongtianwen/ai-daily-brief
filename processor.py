"""
智能推理 Agent (processor.py)

基于 Agentic Workflow 设计哲学:
- 原子技能: 纯函数封装，无状态，独立运行
- 强健错误处理: 单个步骤失败不影响整体流程

功能:
1. 对抓取的内容进行评分筛选
2. 输出结构化 JSON (Top 5)
"""

import os
import json
from dataclasses import asdict
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from loguru import logger

from scraper import TechItem


class ContentProcessor:
    """
    内容智能筛选处理器
    
    这是一个原子技能类:
    - 无状态: 每次调用独立处理
    - 纯函数: 输入 TechItem 列表，输出筛选后的 Top 5
    """
    
    def __init__(self):
        """
        初始化处理器
        """
        pass
    
    def _calculate_score(self, item: TechItem) -> float:
        """
        计算单个技术新闻的评分
        
        Args:
            item: 技术新闻条目
            
        Returns:
            评分 (0-10分)
        """
        score = 0.0
        
        # 1. 基础分 (2分)
        score += 2.0
        
        # 2. GitHub Stars 加分 (最高 3分)
        if item.stars:
            stars_score = min(item.stars / 10000, 3.0)
            score += stars_score
        
        # 3. 技术关键词加分 (最高 3分)
        tech_keywords = [
            # AI/ML 基础
            'ai', 'ml', 'machine learning', 'deep learning', 'neural network',
            # 大模型
            'llm', 'gpt', 'large language model', 'transformer', 'foundation model',
            'model release', 'new model', 'model launch', 'announce', 'release', 'launch',
            # 编程和开发
            'python', 'coding', 'programming', 'developer', 'software engineering',
            'code generation', 'ai programming', 'developer tools',
            # 数据标注
            'data annotation', 'data labeling', 'dataset', 'data collection',
            # AI 企业落地
            'enterprise', 'business', 'industry', 'implementation', 'deployment',
            # 技术趋势
            'generative ai', 'ai assistant', 'prompt engineering', 'rag',
            # 中国相关
            'china', 'chinese', '中文', '中国', '国产化', '国内'
        ]
        
        text_to_check = (item.title + " " + item.description).lower()
        keyword_matches = 0
        
        for keyword in tech_keywords:
            if keyword in text_to_check:
                keyword_matches += 1
                if keyword_matches >= 3:
                    break
        
        score += keyword_matches
        
        # 4. 时效性加分 (最高 2分)
        if hasattr(item, 'published') and item.published:
            try:
                from datetime import datetime, timedelta
                pub_date = datetime.strptime(item.published, '%Y-%m-%d')
                today = datetime.now().date()
                pub_date_only = pub_date.date()
                days_diff = (today - pub_date_only).days
                
                if days_diff == 0:
                    score += 2.0  # 今天发布
                elif days_diff == 1:
                    score += 1.5  # 昨天发布
                elif days_diff == 2:
                    score += 1.0  # 前天发布
                elif days_diff <= 7:
                    score += 0.5  # 一周内
            except:
                pass
        
        # 5. 来源加权
        source_weights = {
            'github': 1.15,
            'huggingface': 1.15,
            'arxiv': 1.10,
            'baidu_dev': 0.95,
            'aliyun_dev': 0.95,
            'tencent_cloud': 0.95,
            'huawei_cloud': 0.95,
            'csdn': 0.90,
            'jiqizhixin': 1.05,  # 机器之心
            'leiphone': 1.00      # 雷锋网
        }
        
        weight = source_weights.get(item.source, 1.0)
        score *= weight
        
        return min(score, 10.0)  # 最高10分
    
    def process(self, items: List[TechItem]) -> List[Dict[str, Any]]:
        """
        处理内容列表，返回筛选后的 Top 5
        
        这是一个原子技能函数:
        - 输入: TechItem 列表
        - 输出: 包含评分和理由的 Top 5 字典列表
        - 错误处理: 任何失败都返回空列表，不崩溃
        
        Args:
            items: 从 scraper 获取的技术新闻列表
            
        Returns:
            Top 5 筛选结果，每个包含 index, title, source, score, reason
        """
        if not items:
            logger.warning("[Processor] 输入内容为空")
            return []
        
        if len(items) <= 5:
            # 如果内容少于5条，直接返回
            logger.info(f"[Processor] 内容只有 {len(items)} 条，跳过评分")
            return [
                {
                    "index": i,
                    "title": item.title,
                    "source": item.source,
                    "score": 7.0,  # 默认分数
                    "reason": "精选内容"
                }
                for i, item in enumerate(items)
            ]
        
        logger.info(f"[Processor] 开始处理 {len(items)} 条内容")
        
        # 计算每条内容的评分
        scored_items = []
        for i, item in enumerate(items):
            try:
                score = self._calculate_score(item)
                scored_items.append((score, i, item))
            except Exception as e:
                logger.warning(f"[Processor] 计算评分失败: {e}")
                continue
        
        # 按评分排序，取前5
        scored_items.sort(reverse=True, key=lambda x: x[0])
        top_items = []
        
        for i, (score, original_index, item) in enumerate(scored_items[:5], 1):
            # 生成评分理由
            reasons = []
            if item.stars and item.stars > 5000:
                reasons.append(f"GitHub Stars: {item.stars}")
            if 'ai' in (item.title + item.description).lower():
                reasons.append("AI相关内容")
            if 'python' in (item.title + item.description).lower():
                reasons.append("Python相关内容")
            if not reasons:
                reasons.append("技术价值高")
            
            top_items.append({
                "index": original_index,
                "title": item.title,
                "source": item.source,
                "score": round(score, 1),
                "reason": ", ".join(reasons[:2])
            })
        
        logger.info(f"[Processor] 筛选完成，返回 Top {len(top_items)}")
        return top_items


def select_top_items(items: List[TechItem]) -> List[Dict[str, Any]]:
    """
    便捷函数: 从 TechItem 列表中筛选 Top 5
    
    使用示例:
        from scraper import fetch_all_sources
        from processor import select_top_items
        
        items = fetch_all_sources()
        top5 = select_top_items(items)
    
    Args:
        items: TechItem 列表
        
    Returns:
        Top 5 筛选结果
    """
    processor = ContentProcessor()
    return processor.process(items)


if __name__ == "__main__":
    # 测试运行
    from dotenv import load_dotenv
    load_dotenv()
    
    # 模拟测试数据
    test_items = [
        TechItem(
            source="github",
            title="microsoft/autogen",
            url="https://github.com/microsoft/autogen",
            description="A programming framework for building AI agents",
            stars=25000
        ),
        TechItem(
            source="github", 
            title="langchain-ai/langchain",
            url="https://github.com/langchain-ai/langchain",
            description="Building applications with LLMs",
            stars=85000
        ),
        TechItem(
            source="arxiv",
            title="Attention Is All You Need",
            url="https://arxiv.org/abs/1706.03762",
            description="The Transformer architecture paper",
            author="Vaswani et al."
        ),
        TechItem(
            source="huggingface",
            title="Llama 3: The Next Generation",
            url="https://huggingface.co/papers/...",
            description="Meta's latest open source LLM"
        ),
        TechItem(
            source="github",
            title="openai/swarm",
            url="https://github.com/openai/swarm",
            description="A lightweight multi-agent orchestration framework",
            stars=15000
        ),
        TechItem(
            source="arxiv",
            title="Tool Learning for Large Language Models",
            url="https://arxiv.org/abs/...",
            description="A survey on tool learning with LLMs"
        ),
    ]
    
    logger.info("=" * 50)
    logger.info("开始测试内容筛选")
    logger.info("=" * 50)
    
    try:
        processor = ContentProcessor()
        top5 = processor.process(test_items)
        
        logger.info("\n筛选结果:")
        for item in top5:
            logger.info(f"\n[{item['score']:.1f}] {item['title']}")
            logger.info(f"来源: {item['source']}")
            logger.info(f"理由: {item['reason']}")
            logger.info("-" * 50)
    except Exception as e:
        logger.error(f"测试失败: {e}")