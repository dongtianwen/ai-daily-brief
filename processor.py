"""
智能推理 Agent (processor.py)

基于 Agentic Workflow 设计哲学:
- 原子技能: 纯函数封装，调用 GLM-5 进行评分筛选
- 熔断机制: JSON 解析失败时直接丢弃该批次，不报错

功能:
1. 调用 GLM-5 对抓取的内容进行 0-10 分评分
2. Agent/Multi-Agent/Tool Use 关键词权重 ×1.5
3. 输出结构化 JSON (Top 5)
"""

import os
import json
from dataclasses import asdict
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from zhipuai import ZhipuAI
from openai import OpenAI
from loguru import logger

from scraper import TechItem


# GLM-5 评分提示词模板
SCORING_PROMPT = """你是一位专业的AI技术内容策展人。请对以下技术新闻进行评分和筛选。

评分标准 (0-10分):
- 创新性: 技术或方法的创新程度
- 实用性: 对开发者和研究者的实际价值
- 时效性: 内容的时效性和热度
- 影响力: 在社区中的潜在影响力

特殊权重规则:
- 如果内容涉及 "LLM", "大模型", "GPT", "Transformer" 等，分数乘以 1.2 倍
- 如果内容涉及 "Python", "编程", "Code Generation", "AI Programming" 等，分数乘以 1.2 倍
- 如果内容涉及 "Data Annotation", "数据标注", "Dataset", "数据采集" 等，分数乘以 1.2 倍
- 如果内容涉及 "Enterprise", "企业", "Business", "落地" 等，分数乘以 1.2 倍
- 如果内容涉及 "NLP", "自然语言", "Text Generation" 等，分数乘以 1.2 倍

你的任务是:
1. 对每条内容进行评分 (0-10分)
2. 选出分数最高的 Top 5
3. 返回严格的 JSON 格式

输入数据:
{items_json}

请返回以下格式的 JSON (不要包含任何其他文字):
{{
    "top_items": [
        {{
            "index": 0,
            "title": "项目标题",
            "source": "github/huggingface/arxiv",
            "score": 8.5,
            "reason": "简要说明为什么这个项目值得关注的理由"
        }}
    ]
}}
"""


class ContentProcessor:
    """
    内容智能筛选处理器
    
    这是一个原子技能类:
    - 无状态: 每次调用独立处理
    - 纯函数: 输入 TechItem 列表，输出筛选后的 Top 5
    """
    
    def __init__(self, api_key: str = None):
        """
        初始化处理器
        
        Args:
            api_key: 智谱 AI API Key，如果不提供则从环境变量读取
        """
        self.api_key = api_key or os.getenv('ZHIPUAI_API_KEY')
        if not self.api_key:
            raise ValueError("ZHIPUAI_API_KEY 未设置")
        
        self.client = ZhipuAI(api_key=self.api_key)
        self.primary_model = "glm-5"  # 主模型
        self.fallback_models = ["glm-4.7"]  # GLM降级模型列表
        self.current_model = self.primary_model  # 当前使用的模型
        self.degraded = False  # 是否已降级
        self.degradation_start_time = None  # 降级开始时间
        
        # NVIDIA备用模型配置
        self.nvidia_api_key = os.getenv('NVIDIA_API_KEY')
        self.nvidia_base_url = os.getenv('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')
        self.nvidia_model = "deepseek-ai/deepseek-v3.2"
        
        if self.nvidia_api_key:
            self.nvidia_client = OpenAI(
                base_url=self.nvidia_base_url,
                api_key=self.nvidia_api_key
            )
        else:
            self.nvidia_client = None
    
    def _prepare_items_json(self, items: List[TechItem]) -> str:
        """将 TechItem 列表转换为 JSON 字符串供 GLM 处理"""
        items_data = []
        for i, item in enumerate(items):
            items_data.append({
                "index": i,
                "source": item.source,
                "title": item.title,
                "description": item.description[:300],  # 限制长度
                "url": item.url,
                "stars": item.stars,
                "author": item.author
            })
        return json.dumps(items_data, ensure_ascii=False, indent=2)
    
    def _call_glm_for_scoring(self, items_json: str) -> Dict[str, Any]:
        """
        调用 GLM 进行评分，支持自动降级
        
        Args:
            items_json: 待评分内容的 JSON 字符串
            
        Returns:
            GLM 返回的 JSON 解析结果
        """
        prompt = SCORING_PROMPT.format(items_json=items_json)
        
        for model in [self.current_model] + self.fallback_models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的AI技术内容策展助手，只返回JSON格式数据。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                
                content = response.choices[0].message.content
                
                if not content or not content.strip():
                    logger.warning(f"[GLM] 模型 {model} 返回空内容")
                    if model != self.fallback_models[-1]:
                        logger.warning(f"[GLM] 尝试降级到下一个模型...")
                        continue
                    else:
                        return {"top_items": []}
                
                if model != self.primary_model and not self.degraded:
                    self._log_degradation_event(model, "GLM")
                
                logger.info(f"[GLM] 成功获取评分响应 (模型: {model})")
                return self._parse_glm_response(content)
                
            except Exception as e:
                error_msg = str(e)
                should_fallback = self._should_fallback(error_msg)
                
                if should_fallback and model != self.fallback_models[-1]:
                    logger.warning(f"[GLM] 模型 {model} 调用失败: {e}")
                    logger.warning(f"[GLM] 尝试降级到下一个模型...")
                    continue
                elif should_fallback:
                    logger.error(f"[GLM] 所有GLM模型均失败: {e}")
                    # 尝试NVIDIA备用模型
                    return self._call_nvidia_for_scoring(items_json)
                else:
                    logger.error(f"[GLM] API 调用失败: {e}")
                    return {"top_items": []}
        
        return {"top_items": []}
    
    def _call_nvidia_for_scoring(self, items_json: str) -> Dict[str, Any]:
        """
        调用 NVIDIA NIM API 进行评分 (备用)
        
        Args:
            items_json: 待评分内容的 JSON 字符串
            
        Returns:
            NVIDIA 返回的 JSON 解析结果
        """
        if not self.nvidia_client:
            logger.warning("[NVIDIA] NVIDIA API 未配置，跳过备用模型")
            return {"top_items": []}
        
        prompt = SCORING_PROMPT.format(items_json=items_json)
        
        try:
            logger.info(f"[NVIDIA] 尝试使用 NVIDIA 备用模型: {self.nvidia_model}")
            
            response = self.nvidia_client.chat.completions.create(
                model=self.nvidia_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的AI技术内容策展助手，只返回JSON格式数据。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            if not content or not content.strip():
                logger.warning("[NVIDIA] 模型返回空内容")
                return {"top_items": []}
            
            self._log_degradation_event(self.nvidia_model, "NVIDIA")
            
            logger.info(f"[NVIDIA] 成功获取评分响应 (模型: {self.nvidia_model})")
            return self._parse_glm_response(content)
            
        except Exception as e:
            logger.error(f"[NVIDIA] API 调用失败: {e}")
            return {"top_items": []}
    
    def _should_fallback(self, error_msg: str) -> bool:
        """
        判断是否应该降级
        
        Args:
            error_msg: 错误信息
            
        Returns:
            是否应该降级
        """
        fallback_indicators = [
            "429",  # 速率限制
            "500",  # 服务器错误
            "502",  # 网关错误
            "503",  # 服务不可用
            "504",  # 网关超时
            "timeout",  # 超时
            "connection",  # 连接错误
            "service unavailable",  # 服务不可用
            "rate limit",  # 速率限制
            "quota",  # 配额限制
        ]
        
        error_msg_lower = error_msg.lower()
        return any(indicator.lower() in error_msg_lower for indicator in fallback_indicators)
    
    def _log_degradation_event(self, new_model: str, provider: str = "GLM"):
        """
        记录降级事件
        
        Args:
            new_model: 降级后的模型
            provider: 模型提供商
        """
        self.degraded = True
        self.current_model = new_model
        self.degradation_start_time = datetime.now()
        
        logger.warning("=" * 60)
        logger.warning(f"[降级] 模型降级事件")
        logger.warning(f"[降级] 时间: {self.degradation_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.warning(f"[降级] 提供商: {provider}")
        logger.warning(f"[降级] 新模型: {new_model}")
        logger.warning("=" * 60)
    
    def _parse_glm_response(self, content: str) -> Dict[str, Any]:
        """
        解析 GLM 返回的内容为 JSON
        
        熔断机制:
        - 如果 JSON 解析失败，返回空结果，不抛出异常
        """
        try:
            # 尝试直接解析
            result = json.loads(content)
            if "top_items" in result:
                return result
        except json.JSONDecodeError:
            pass
        
        # 尝试从 Markdown 代码块中提取 JSON
        try:
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
            if json_match:
                result = json.loads(json_match.group(1))
                if "top_items" in result:
                    return result
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # 尝试从文本中提取 JSON 对象
        try:
            json_match = re.search(r'\{[\s\S]*"top_items"[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group(0))
                if "top_items" in result:
                    return result
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # 熔断: 所有解析尝试都失败，返回空结果
        logger.warning(f"[Processor] JSON 解析失败，触发熔断机制。原始响应:\n{content[:1000]}...")
        logger.debug(f"[Processor] 完整响应:\n{content}")
        
        # 保存完整响应到文件用于调试
        try:
            import os
            debug_dir = Path("output/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_file = debug_dir / f"model_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"模型响应内容:\n{'='*50}\n\n{content}\n\n{'='*50}\n")
            logger.info(f"[Processor] 完整响应已保存到: {debug_file}")
        except Exception as e:
            logger.warning(f"[Processor] 保存调试文件失败: {e}")
        
        return {"top_items": []}
    
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
        
        # 准备输入数据
        items_json = self._prepare_items_json(items)
        
        # 调用模型评分
        result = self._call_glm_for_scoring(items_json)
        
        top_items = result.get("top_items", [])
        
        if not top_items:
            logger.warning("[Processor] 模型返回空结果，使用备用策略")
            # 备用策略: 按 stars 排序取前5
            sorted_items = sorted(
                enumerate(items), 
                key=lambda x: x[1].stars or 0, 
                reverse=True
            )[:5]
            top_items = [
                {
                    "index": idx,
                    "title": item.title,
                    "source": item.source,
                    "score": 6.0 + (item.stars or 0) / 10000,
                    "reason": f"GitHub Stars: {item.stars or 'N/A'}"
                }
                for idx, item in sorted_items
            ]
        
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