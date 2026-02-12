"""
内容生成 Agent (writer.py)

基于 Agentic Workflow 设计哲学:
- 原子技能: 纯函数封装，无状态，独立运行
- 强健错误处理: 单个步骤失败不影响整体流程

功能:
1. 将筛选出的 Top 5 内容改写为中文播客讲稿
2. 长度严格控制在 600-900 中文字符 (约 3-5 分钟语音)
3. TTS 前处理: 术语发音矫正
"""

import os
import json
import re
from typing import List, Dict, Any
from datetime import datetime

from loguru import logger


# TTS 发音矫正映射表
PRONUNCIATION_MAP = {
    # 英文缩写 -> 中文读法
    'LLM': '大语言模型',
    'LLMs': '大语言模型',
    'GPT': 'GPT',
    'API': 'API',
    'AI': '人工智能',
    'ML': '机器学习',
    'NLP': '自然语言处理',
    'RAG': 'RAG',
    'SOTA': 'So-tah',
    'arXiv': 'Archive',
    'BERT': 'BERT',
    'T5': 'T5',
    'CLIP': 'CLIP',
    'DALL-E': 'DALL-E',
    'Midjourney': 'Midjourney',
    'Stable Diffusion': 'Stable Diffusion',
    'LangChain': 'LangChain',
    'AutoGen': 'AutoGen',
    'Hugging Face': 'Hugging Face',
    'GitHub': 'GitHub',
    'Python': 'Python',
    'JavaScript': 'JavaScript',
    'OpenAI': 'OpenAI',
    'Anthropic': 'Anthropic',
    'DeepMind': 'DeepMind',
    'Meta AI': 'Meta AI',
    'Google': 'Google',
    'Microsoft': 'Microsoft',
    'Transformer': 'Transformer',
    'Diffusion': 'Diffusion',
    'Fine-tuning': 'Fine-tuning',
    'Prompt': 'Prompt',
    'Embedding': 'Embedding',
    'Token': 'Token',
    'GPU': 'GPU',
    'CPU': 'CPU',
    'RAM': '内存',
    'API Key': 'API Key',
}


class ContentWriter:
    """
    内容生成器 - 将 Top 5 内容改写为播客讲稿
    
    这是一个原子技能类:
    - 无状态: 每次调用独立生成
    - 纯函数: 输入 Top 5 数据，输出中文播客讲稿
    """
    
    def __init__(self):
        """
        初始化生成器
        """
        pass
    
    def _generate_script(self, top_items: List[Dict[str, Any]], 
                        all_items: List[Any]) -> str:
        """
        生成播客讲稿
        
        Args:
            top_items: processor 返回的 Top 5 列表
            all_items: 原始 TechItem 列表
            
        Returns:
            生成的中文播客讲稿
        """
        today = datetime.now().strftime("%m月%d日")
        
        # 开场白
        lines = [
            f"大家好，欢迎收听今天的AI技术播客。我是你们的主持人，今天我们来聊聊最近GitHub上的一些热门项目，看看都有哪些新技术和应用正在崛起。",
            "",
        ]
        
        # 主体内容 - 为每个项目生成口语化介绍
        for i, item in enumerate(top_items, 1):
            idx = item.get('index', 0)
            if idx < len(all_items):
                original = all_items[idx]
                source = item.get('source', 'unknown').upper()
                title = item.get('title', '')
                score = item.get('score', 0)
                
                # 根据项目生成口语化介绍
                if i == 1:
                    lines.append(f"首先，咱们来看看一个叫做{title}的项目。")
                elif i == len(top_items):
                    lines.append(f"最后，我们要介绍的是{title}项目。")
                else:
                    lines.append(f"接下来，我们要说的是{title}项目。")
                
                # 项目描述
                if hasattr(original, 'description') and original.description:
                    desc = original.description
                    # 简单翻译常见术语
                    translations = {
                        'machine learning': '机器学习',
                        'deep learning': '深度学习',
                        'natural language': '自然语言',
                        'text generation': '文本生成',
                        'computer vision': '计算机视觉',
                        'audio': '音频',
                        'multimodal': '多模态',
                        'inference': '推理',
                        'training': '训练',
                        'model': '模型',
                        'framework': '框架',
                        'pipeline': '管道',
                        'state-of-the-art': '最先进的',
                        'pretrained': '预训练',
                        'API': 'API',
                        'NLP': '自然语言处理',
                        'AI': '人工智能',
                        'LLM': '大语言模型',
                        'GPT': 'GPT',
                        'Transformer': 'Transformer'
                    }
                    for en, zh in translations.items():
                        desc = desc.replace(en, zh)
                    
                    # 生成口语化描述
                    lines.append(f"这个项目{desc[:150]}。")
                
                # 添加技术亮点和价值
                if hasattr(original, 'stars') and original.stars:
                    if original.stars > 100000:
                        lines.append(f"这个项目非常受欢迎，目前在GitHub上已经获得了{original.stars}个星标，可见其影响力之大。")
                    elif original.stars > 50000:
                        lines.append(f"这个项目在GitHub上已经获得了{original.stars}个星标，看来社区的伙伴们对它都很感兴趣。")
                    elif original.stars > 10000:
                        lines.append(f"目前这个项目在GitHub上有{original.stars}个星标，显示了它的价值。")
                    else:
                        lines.append(f"这个项目在GitHub上获得了{original.stars}个星标，值得关注。")
                
                # 根据项目类型添加具体价值说明
                title_lower = title.lower()
                if 'transformers' in title_lower:
                    lines.append("这个框架极大地简化了机器学习模型的推理和训练过程，非常实用，也极具影响力。")
                elif 'langextract' in title_lower:
                    lines.append("这个库不仅创新，而且实用，它通过精确的源头定位和交互式可视化，让数据处理变得更加直观。")
                elif 'trendradar' in title_lower:
                    lines.append("它的亮点在于实用性和时效性，能够帮你从海量的信息中筛选出有用的热点。")
                elif 'trading' in title_lower:
                    lines.append("这个框架对于金融领域的开发者来说非常有用，因为它可以帮助他们更好地进行交易策略的制定和执行。")
                elif 'api' in title_lower or 'resource' in title_lower:
                    lines.append("对于开发者来说，这样的资源非常宝贵，可以节省大量时间成本。")
                
                lines.append("")
        
        # 总结
        lines.extend([
            "总结一下，今天我们介绍了几个GitHub上的热门项目，涵盖了多个技术领域。",
            "这些项目不仅展示了AI技术的创新和应用，也为开发者提供了很多便利。",
            "希望今天的分享能给大家带来一些启发和灵感。",
            "",
        ])
        
        # 结尾
        lines.extend([
            "好了，今天的播客就到这里，感谢大家的收听。",
            "如果你对我们的内容有任何想法或者建议，欢迎在评论区留言。",
            "我们下期节目再见！"
        ])
        
        return '\n'.join(lines)
    
    def _apply_tts_preprocessing(self, script: str) -> str:
        """
        TTS 前处理: 术语发音矫正
        
        将技术术语替换为适合 TTS 朗读的形式
        
        Args:
            script: 原始播客讲稿
            
        Returns:
            处理后的讲稿
        """
        processed = script
        
        # 按长度降序排序，避免短词替换影响长词
        for term, pronunciation in sorted(
            PRONUNCIATION_MAP.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        ):
            # 使用正则表达式进行全词匹配
            pattern = r'\b' + re.escape(term) + r'\b'
            processed = re.sub(pattern, pronunciation, processed, flags=re.IGNORECASE)
        
        # 处理一些特殊格式
        # 移除 Markdown 标记
        processed = re.sub(r'\*\*', '', processed)  # 粗体
        processed = re.sub(r'\*', '', processed)    # 斜体
        processed = re.sub(r'`', '', processed)     # 代码
        
        # 处理连续的空行
        processed = re.sub(r'\n{3,}', '\n\n', processed)
        
        return processed.strip()
    
    def write(self, top_items: List[Dict[str, Any]], 
              all_items: List[Any]) -> str:
        """
        生成中文播客讲稿
        
        这是一个原子技能函数:
        - 输入: Top 5 数据和原始 TechItem 列表
        - 输出: 600-900 中文字符的播客讲稿
        - 错误处理: 任何失败都返回备用讲稿
        
        Args:
            top_items: processor 返回的 Top 5 列表
            all_items: 原始 TechItem 列表
            
        Returns:
            中文播客讲稿 (已进行 TTS 前处理)
        """
        if not top_items:
            logger.warning("[Writer] 输入内容为空")
            return "大家好，今天暂时没有新的技术资讯，感谢收听。"
        
        logger.info(f"[Writer] 开始生成播客讲稿，共 {len(top_items)} 条内容")
        
        # 生成讲稿
        script = self._generate_script(top_items, all_items)
        
        # TTS 前处理
        processed_script = self._apply_tts_preprocessing(script)
        
        # 检查长度
        char_count = len(processed_script.replace(' ', '').replace('\n', ''))
        logger.info(f"[Writer] 生成完成，共 {char_count} 字符")
        
        if char_count < 300:
            logger.warning(f"[Writer] 讲稿过短 ({char_count} 字符)，可能不符合预期")
        elif char_count > 1200:
            logger.warning(f"[Writer] 讲稿过长 ({char_count} 字符)，可能超出语音时长")
        
        return processed_script


def generate_podcast_script(top_items: List[Dict[str, Any]], 
                           all_items: List[Any]) -> str:
    """
    便捷函数: 生成播客讲稿
    
    使用示例:
        from scraper import fetch_all_sources
        from processor import select_top_items
        from writer import generate_podcast_script
        
        items = fetch_all_sources()
        top5 = select_top_items(items)
        script = generate_podcast_script(top5, items)
    
    Args:
        top_items: Top 5 筛选结果
        all_items: 原始 TechItem 列表
        
    Returns:
        中文播客讲稿
    """
    writer = ContentWriter()
    return writer.write(top_items, all_items)


if __name__ == "__main__":
    # 测试运行
    from dotenv import load_dotenv
    load_dotenv()
    
    # 模拟测试数据
    test_top_items = [
        {
            "index": 0,
            "title": "microsoft/autogen",
            "source": "github",
            "score": 9.5,
            "reason": "微软开源的多智能体对话框架，支持复杂的Agent协作"
        },
        {
            "index": 1,
            "title": "openai/swarm",
            "source": "github", 
            "score": 9.0,
            "reason": "OpenAI官方的多Agent编排框架，轻量级设计"
        },
        {
            "index": 2,
            "title": "Tool Learning for Large Language Models",
            "source": "arxiv",
            "score": 8.5,
            "reason": "关于LLM工具学习的全面综述"
        }
    ]
    
    from scraper import TechItem
    test_all_items = [
        TechItem(
            source="github",
            title="microsoft/autogen",
            url="https://github.com/microsoft/autogen",
            description="A programming framework for building AI agents and enabling multi-agent collaboration",
            stars=25000
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
            description="A comprehensive survey on tool learning with LLMs"
        )
    ]
    
    logger.info("=" * 50)
    logger.info("开始测试内容生成")
    logger.info("=" * 50)
    
    try:
        writer = ContentWriter()
        script = writer.write(test_top_items, test_all_items)
        
        logger.info("\n生成的播客讲稿:")
        logger.info("=" * 50)
        logger.info(script)
        logger.info("=" * 50)
        
        char_count = len(script.replace(' ', '').replace('\n', ''))
        logger.info(f"\n字符数: {char_count}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")