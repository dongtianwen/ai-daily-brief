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
    
    def _prepare_items_text(self, top_items: List[Dict[str, Any]], 
                           all_items: List[Any]) -> str:
        """
        将 Top 5 数据转换为文本格式
        
        Args:
            top_items: processor 返回的 Top 5 列表
            all_items: 原始 TechItem 列表
            
        Returns:
            格式化的文本
        """
        lines = []
        for i, item in enumerate(top_items, 1):
            idx = item.get('index', 0)
            if idx < len(all_items):
                original = all_items[idx]
                lines.append(f"\n{i}. 【{item.get('source', 'unknown').upper()}】{item.get('title', '')}")
                if hasattr(original, 'description') and original.description:
                    lines.append(f"   简介: {original.description[:150]}")
                if item.get('reason'):
                    lines.append(f"   亮点: {item['reason']}")
                if hasattr(original, 'stars') and original.stars:
                    lines.append(f"   Stars: {original.stars}")
        
        return '\n'.join(lines)
    
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
        
        # 开场
        lines = [
            f"大家好，欢迎收听AI每日技术简报，今天是{today}。",
            "",
            "在今天的节目中，我们为您精选了几条最值得关注的AI技术资讯：",
            "",
        ]
        
        # 主体内容
        for i, item in enumerate(top_items, 1):
            idx = item.get('index', 0)
            if idx < len(all_items):
                original = all_items[idx]
                source = item.get('source', 'unknown').upper()
                title = item.get('title', '')
                reason = item.get('reason', '')
                score = item.get('score', 0)
                
                # 构建项目介绍
                project_lines = []
                project_lines.append(f"{i}. 【{source}】{title}")
                project_lines.append(f"   评分：{score:.1f}分")
                
                if hasattr(original, 'description') and original.description:
                    # 提取并处理描述
                    desc = original.description[:200]
                    project_lines.append(f"   项目简介：{desc}")
                
                if reason:
                    project_lines.append(f"   技术亮点：{reason}")
                
                if hasattr(original, 'stars') and original.stars:
                    project_lines.append(f"   GitHub Stars：{original.stars}")
                
                if hasattr(original, 'url') and original.url:
                    project_lines.append(f"   链接：{original.url}")
                
                # 添加技术分析
                project_lines.append("   技术分析：")
                if 'ai' in (title + str(original.description)).lower():
                    project_lines.append("   - 该项目属于人工智能领域，具有较高的技术创新性")
                if 'github' in source.lower():
                    project_lines.append("   - 来自GitHub平台，社区活跃度高，值得关注")
                if 'arxiv' in source.lower():
                    project_lines.append("   - 来自学术论文，代表最新研究成果")
                
                project_lines.append("")
                lines.extend(project_lines)
        
        # 技术趋势分析
        lines.extend([
            "【技术趋势分析】",
            "",
        ])
        
        # 分析当天的技术趋势
        trends = []
        for item in top_items:
            idx = item.get('index', 0)
            if idx < len(all_items):
                original = all_items[idx]
                text = (item.get('title', '') + " " + str(original.description)).lower()
                if 'llm' in text or '大模型' in text:
                    trends.append("大模型技术")
                elif 'ai programming' in text or '代码生成' in text:
                    trends.append("AI编程工具")
                elif 'data' in text and ('annotation' in text or 'labeling' in text):
                    trends.append("数据标注技术")
                elif 'enterprise' in text or '企业' in text:
                    trends.append("AI企业落地")
        
        # 去重并统计
        unique_trends = list(set(trends))
        if unique_trends:
            lines.append(f"根据今日精选内容，我们可以看到{', '.join(unique_trends)}等技术领域正在成为热点。")
        else:
            lines.append("今日内容涵盖了多个AI技术领域，展现了行业的多元化发展趋势。")
        
        lines.append("")
        
        # 结尾
        lines.extend([
            "【结语】",
            "",
            "以上就是今天的AI技术简报全部内容，感谢大家的收听。",
            "我们每天都会为您精选最有价值的AI技术资讯，",
            "关注技术前沿动态，把握行业发展趋势，",
            "欢迎持续关注我们的节目，我们明天再见！"
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