"""
内容生成 Agent (writer.py)

基于 Agentic Workflow 设计哲学:
- 原子技能: 纯函数封装，将 Top 5 内容改写为中文播客讲稿
- 无状态: 每次调用独立生成

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

from zhipuai import ZhipuAI
from loguru import logger


# 文案生成提示词模板
WRITING_PROMPT = """你是一位专业的AI技术播客主持人。请将以下精选的AI技术新闻改写成一段自然、流畅的中文播客讲稿。

要求:
1. 语言风格: 专业但不晦涩，像朋友聊天一样自然，口语化表达
2. 结构: 
   - 开场问候 (简短)
   - 正文: 逐一详细介绍每个项目/论文，突出技术亮点、创新点和应用价值
   - 结尾 (简短总结和展望)
3. 长度: 必须控制在 800-1000 个中文字符 (约 4-6 分钟语音)
4. 内容要求: 
   - 详细展开每个项目的技术细节、开发意义和商业价值
   - 加入适当的背景介绍，帮助听众理解技术背景
   - 重点关注 AI 编程、大模型、数据标注、AI 企业落地等领域
   - 每个项目至少 150 字的详细介绍
5. 格式: 纯文本，不需要 Markdown 标记
6. 发音处理: 保留以下术语的原始写法，系统会自动处理发音
   - LLM -> 大语言模型
   - SOTA -> So-tah
   - arXiv -> Archive

今日精选内容:
{items_text}

请直接输出播客讲稿文本，确保长度达到 800 字以上，详细介绍每个项目，不要包含任何其他格式或标记。
"""


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
    
    def __init__(self, api_key: str = None):
        """
        初始化生成器
        
        Args:
            api_key: 智谱 AI API Key，如果不提供则从环境变量读取
        """
        self.api_key = api_key or os.getenv('ZHIPUAI_API_KEY')
        if not self.api_key:
            raise ValueError("ZHIPUAI_API_KEY 未设置")
        
        self.client = ZhipuAI(api_key=self.api_key)
        self.model = "glm-4"
    
    def _prepare_items_text(self, top_items: List[Dict[str, Any]], 
                           all_items: List[Any]) -> str:
        """
        将 Top 5 数据转换为文本格式供 GLM 处理
        
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
    
    def _call_glm_for_writing(self, items_text: str) -> str:
        """
        调用 GLM-4 生成播客讲稿
        
        Args:
            items_text: 格式化的内容文本
            
        Returns:
            生成的中文播客讲稿
        """
        prompt = WRITING_PROMPT.format(items_text=items_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的AI技术播客主持人，擅长用通俗易懂的语言讲解技术内容。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # 适中温度以获得自然流畅的文本
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            logger.info(f"[Writer] 成功生成播客讲稿")
            return content.strip()
            
        except Exception as e:
            logger.error(f"[Writer] GLM API 调用失败: {e}")
            return self._generate_fallback_script(items_text)
    
    def _generate_fallback_script(self, items_text: str) -> str:
        """
        备用策略: 当 GLM 调用失败时生成简单的讲稿
        
        Args:
            items_text: 格式化的内容文本
            
        Returns:
            简单的中文讲稿
        """
        today = datetime.now().strftime("%m月%d日")
        lines = [
            f"大家好，欢迎收听AI每日技术简报，今天是{today}。",
            "",
            "今天为大家精选了以下几条AI技术资讯：",
            "",
        ]
        
        # 简单解析 items_text
        for line in items_text.split('\n'):
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                lines.append(line.strip())
        
        lines.extend([
            "",
            "以上就是今天的内容，感谢收听，我们明天再见。"
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
        
        # 准备输入数据
        items_text = self._prepare_items_text(top_items, all_items)
        
        # 调用 GLM 生成讲稿
        script = self._call_glm_for_writing(items_text)
        
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
