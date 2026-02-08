"""
语音合成技能 (audio.py)

基于 Agentic Workflow 设计哲学:
- 原子技能: 纯函数封装，调用 edge-tts 生成语音
- 无状态: 每次调用独立生成音频文件

功能:
1. 使用 edge-tts (微软 Edge 免费高品质接口) 生成语音
2. 支持中文语音合成
3. DEBUG 模式跳过 TTS 以节省成本
"""

import os
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

import edge_tts
from loguru import logger


# 默认语音配置
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"  # 中文女声，自然流畅
DEFAULT_RATE = "+0%"   # 语速
DEFAULT_VOLUME = "+0%"  # 音量


class TTSEngine:
    """
    语音合成引擎
    
    这是一个原子技能类:
    - 无状态: 每次调用独立生成音频
    - 纯函数: 输入文本，输出音频文件路径
    """
    
    def __init__(self, 
                 voice: str = None,
                 rate: str = None,
                 volume: str = None,
                 output_dir: str = None):
        """
        初始化 TTS 引擎
        
        Args:
            voice: 语音角色，默认 zh-CN-XiaoxiaoNeural
            rate: 语速调整，如 "+10%" 或 "-10%"
            volume: 音量调整，如 "+10%" 或 "-10%"
            output_dir: 音频文件输出目录
        """
        self.voice = voice or DEFAULT_VOICE
        self.rate = rate or DEFAULT_RATE
        self.volume = volume or DEFAULT_VOLUME
        self.output_dir = Path(output_dir or "output/audio")
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_filename(self) -> str:
        """生成音频文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"daily_brief_{timestamp}.mp3"
    
    async def _synthesize_async(self, text: str, output_path: Path) -> bool:
        """
        异步合成语音
        
        Args:
            text: 要合成的文本
            output_path: 输出文件路径
            
        Returns:
            是否成功
        """
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume
            )
            
            await communicate.save(str(output_path))
            logger.info(f"[TTS] 语音合成成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"[TTS] 语音合成失败: {e}")
            return False
    
    def synthesize(self, text: str, output_path: str = None, save_text: bool = True) -> Optional[str]:
        """
        将文本合成为语音
        
        这是一个原子技能函数:
        - 输入: 中文播客讲稿文本
        - 输出: MP3 音频文件路径
        - 错误处理: 失败返回 None
        
        Args:
            text: 要合成的中文文本
            output_path: 可选，指定输出路径
            save_text: 是否保存文字内容到文件
            
        Returns:
            生成的音频文件路径，失败返回 None
        """
        if not text or not text.strip():
            logger.warning("[TTS] 输入文本为空")
            return None
        
        # 确定输出路径
        if output_path:
            output_file = Path(output_path)
        else:
            output_file = self.output_dir / self._generate_filename()
        
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[TTS] 开始合成语音，文本长度: {len(text)} 字符")
        logger.info(f"[TTS] 使用语音: {self.voice}")
        
        # 运行异步合成
        try:
            success = asyncio.run(self._synthesize_async(text, output_file))
            
            if success and output_file.exists():
                file_size = output_file.stat().st_size
                logger.info(f"[TTS] 音频文件生成成功: {file_size} bytes")
                
                # 保存文字内容到文件
                if save_text:
                    text_file = output_file.with_suffix('.txt')
                    try:
                        with open(text_file, 'w', encoding='utf-8') as f:
                            f.write(text)
                        logger.info(f"[TTS] 文字内容已保存: {text_file}")
                    except Exception as e:
                        logger.warning(f"[TTS] 保存文字内容失败: {e}")
                
                return str(output_file)
            else:
                logger.error("[TTS] 音频文件生成失败")
                return None
                
        except Exception as e:
            logger.error(f"[TTS] 合成过程异常: {e}")
            return None
    
    def get_available_voices(self) -> list:
        """
        获取可用的中文语音列表
        
        Returns:
            可用语音角色列表
        """
        # edge-tts 常用中文语音
        chinese_voices = [
            ("zh-CN-XiaoxiaoNeural", "中文女声 - 晓晓 (自然流畅，推荐)"),
            ("zh-CN-YunyangNeural", "中文男声 - 云扬"),
            ("zh-CN-YunxiNeural", "中文男声 - 云希 (年轻)"),
            ("zh-CN-YunjianNeural", "中文男声 - 云健 (新闻播报风格)"),
            ("zh-CN-XiaoyiNeural", "中文女声 - 晓伊"),
            ("zh-CN-XiaochenNeural", "中文女声 - 晓晨"),
            ("zh-TW-HsiaoChenNeural", "台湾女声 - 晓臻"),
            ("zh-TW-YunJheNeural", "台湾男声 - 云哲"),
            ("zh-HK-HiuMaanNeural", "香港女声 - 晓曼"),
        ]
        return chinese_voices


def generate_audio(text: str, 
                   output_path: str = None,
                   voice: str = None,
                   save_text: bool = True) -> Optional[str]:
    """
    便捷函数: 将文本合成为语音
    
    使用示例:
        from writer import generate_podcast_script
        from audio import generate_audio
        
        script = generate_podcast_script(top5, items)
        audio_path = generate_audio(script)
    
    Args:
        text: 要合成的中文文本
        output_path: 可选，指定输出路径
        voice: 可选，指定语音角色
        save_text: 是否保存文字内容到文件
        
    Returns:
        生成的音频文件路径，失败返回 None
    """
    engine = TTSEngine(voice=voice)
    return engine.synthesize(text, output_path, save_text)


def generate_audio_with_metadata(text: str,
                                  top_items: list = None,
                                  output_dir: str = None) -> Optional[dict]:
    """
    生成音频并返回包含元数据的字典
    
    Args:
        text: 要合成的文本
        top_items: 相关的 Top 5 数据，用于生成元数据
        output_dir: 输出目录
        
    Returns:
        包含音频路径和元数据的字典
    """
    engine = TTSEngine(output_dir=output_dir)
    audio_path = engine.synthesize(text)
    
    if not audio_path:
        return None
    
    metadata = {
        "audio_path": audio_path,
        "created_at": datetime.now().isoformat(),
        "voice": engine.voice,
        "text_length": len(text),
        "text_preview": text[:100] + "..." if len(text) > 100 else text
    }
    
    if top_items:
        metadata["items"] = [
            {"title": item.get("title"), "source": item.get("source")}
            for item in top_items
        ]
    
    return metadata


if __name__ == "__main__":
    # 测试运行
    from dotenv import load_dotenv
    load_dotenv()
    
    logger.info("=" * 50)
    logger.info("开始测试语音合成")
    logger.info("=" * 50)
    
    # 显示可用语音
    engine = TTSEngine()
    logger.info("\n可用中文语音:")
    for voice_id, description in engine.get_available_voices():
        logger.info(f"  {voice_id}: {description}")
    
    # 测试文本
    test_text = """
    大家好，欢迎收听AI每日技术简报。
    
    今天为大家介绍几个值得关注的AI项目。
    
    首先是微软开源的AutoGen框架，这是一个多智能体对话框架，
    支持复杂的Agent协作，让多个AI智能体可以协同完成任务。
    
    第二个是OpenAI发布的Swarm，这是一个轻量级的多Agent编排框架，
    设计理念简洁优雅，适合快速原型开发。
    
    以上就是今天的内容，感谢收听，我们明天再见。
    """
    
    logger.info("\n测试文本:")
    logger.info(test_text)
    logger.info("-" * 50)
    
    # 合成语音
    audio_path = engine.synthesize(test_text.strip())
    
    if audio_path:
        logger.info(f"\n音频文件已生成: {audio_path}")
    else:
        logger.error("\n音频生成失败")
