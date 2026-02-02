"""
消息推送技能 (notifier.py)

基于 Agentic Workflow 设计哲学:
- 原子技能: 纯函数封装，调用企业微信 API 推送消息
- 无状态: 每次调用独立推送
- DEBUG 模式: 只打印到控制台，不实际发送

功能:
1. 企业微信自建应用 API 推送
2. 支持发送语音消息 + 图文卡片
3. DEBUG 模式只打印不发送
"""

import os
import json
import base64
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import requests
from loguru import logger


class WeComNotifier:
    """
    企业微信消息推送器
    
    这是一个原子技能类:
    - 无状态: 每次调用独立推送
    - 纯函数: 输入消息内容，输出推送结果
    """
    
    BASE_URL = "https://qyapi.weixin.qq.com/cgi-bin"
    
    def __init__(self,
                 corp_id: str = None,
                 agent_id: str = None,
                 secret: str = None,
                 to_user: str = None):
        """
        初始化企业微信推送器
        
        Args:
            corp_id: 企业ID
            agent_id: 应用 Agent ID
            secret: 应用 Secret
            to_user: 接收消息的用户ID，默认 @all (所有人)
        """
        self.corp_id = corp_id or os.getenv('CORP_ID')
        self.agent_id = agent_id or os.getenv('AGENT_ID')
        self.secret = secret or os.getenv('SECRET')
        self.to_user = to_user or os.getenv('TO_USER', '@all')
        
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
    
    def _get_access_token(self) -> Optional[str]:
        """
        获取企业微信 Access Token
        
        Token 有效期为 2 小时，需要缓存
        """
        # 检查缓存的 token 是否有效
        if self._access_token and self._token_expires_at:
            if datetime.now() < self._token_expires_at:
                return self._access_token
        
        if not all([self.corp_id, self.secret]):
            logger.error("[WeCom] 缺少企业微信配置 (CORP_ID 或 SECRET)")
            return None
        
        url = f"{self.BASE_URL}/gettoken"
        params = {
            "corpid": self.corp_id,
            "corpsecret": self.secret
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("errcode") == 0:
                self._access_token = data["access_token"]
                # 提前 5 分钟过期，避免边界问题
                expires_in = data.get("expires_in", 7200) - 300
                from datetime import timedelta
                self._token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                logger.info("[WeCom] Access Token 获取成功")
                return self._access_token
            else:
                logger.error(f"[WeCom] 获取 Token 失败: {data}")
                return None
                
        except Exception as e:
            logger.error(f"[WeCom] 获取 Token 异常: {e}")
            return None
    
    def _upload_media(self, media_path: str, media_type: str = "voice") -> Optional[str]:
        """
        上传临时素材到企业微信
        
        Args:
            media_path: 媒体文件路径
            media_type: 媒体类型 (voice/voice/image/file)
            
        Returns:
            media_id，用于后续消息发送
        """
        access_token = self._get_access_token()
        if not access_token:
            return None
        
        url = f"{self.BASE_URL}/media/upload"
        params = {
            "access_token": access_token,
            "type": media_type
        }
        
        try:
            with open(media_path, 'rb') as f:
                files = {'media': f}
                response = requests.post(url, params=params, files=files, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data.get("errcode") == 0:
                    media_id = data["media_id"]
                    logger.info(f"[WeCom] 媒体上传成功: {media_id}")
                    return media_id
                else:
                    logger.error(f"[WeCom] 媒体上传失败: {data}")
                    return None
                    
        except Exception as e:
            logger.error(f"[WeCom] 媒体上传异常: {e}")
            return None
    
    def send_text_card(self, title: str, description: str, url: str = None,
                       btntxt: str = "查看详情") -> bool:
        """
        发送图文卡片消息
        
        Args:
            title: 标题
            description: 描述
            url: 点击跳转链接
            btntxt: 按钮文字
            
        Returns:
            是否发送成功
        """
        access_token = self._get_access_token()
        if not access_token:
            return False
        
        api_url = f"{self.BASE_URL}/message/send"
        params = {"access_token": access_token}
        
        data = {
            "touser": self.to_user,
            "msgtype": "textcard",
            "agentid": self.agent_id,
            "textcard": {
                "title": title,
                "description": description,
                "url": url or "https://github.com/trending",
                "btntxt": btntxt
            },
            "safe": 0
        }
        
        try:
            response = requests.post(api_url, params=params, 
                                    json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get("errcode") == 0:
                logger.info("[WeCom] 图文卡片发送成功")
                return True
            else:
                logger.error(f"[WeCom] 图文卡片发送失败: {result}")
                return False
                
        except Exception as e:
            logger.error(f"[WeCom] 图文卡片发送异常: {e}")
            return False
    
    def send_voice(self, media_path: str) -> bool:
        """
        发送语音消息
        
        Args:
            media_path: 语音文件路径 (MP3/AMR 格式)
            
        Returns:
            是否发送成功
        """
        # 上传语音文件获取 media_id
        media_id = self._upload_media(media_path, media_type="voice")
        if not media_id:
            return False
        
        access_token = self._get_access_token()
        if not access_token:
            return False
        
        api_url = f"{self.BASE_URL}/message/send"
        params = {"access_token": access_token}
        
        data = {
            "touser": self.to_user,
            "msgtype": "voice",
            "agentid": self.agent_id,
            "voice": {
                "media_id": media_id
            },
            "safe": 0
        }
        
        try:
            response = requests.post(api_url, params=params,
                                    json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get("errcode") == 0:
                logger.info("[WeCom] 语音消息发送成功")
                return True
            else:
                logger.error(f"[WeCom] 语音消息发送失败: {result}")
                return False
                
        except Exception as e:
            logger.error(f"[WeCom] 语音消息发送异常: {e}")
            return False
    
    def send_text(self, content: str) -> bool:
        """
        发送文本消息
        
        Args:
            content: 文本内容
            
        Returns:
            是否发送成功
        """
        access_token = self._get_access_token()
        if not access_token:
            return False
        
        api_url = f"{self.BASE_URL}/message/send"
        params = {"access_token": access_token}
        
        data = {
            "touser": self.to_user,
            "msgtype": "text",
            "agentid": self.agent_id,
            "text": {
                "content": content
            },
            "safe": 0
        }
        
        try:
            response = requests.post(api_url, params=params,
                                    json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get("errcode") == 0:
                logger.info("[WeCom] 文本消息发送成功")
                return True
            else:
                logger.error(f"[WeCom] 文本消息发送失败: {result}")
                return False
                
        except Exception as e:
            logger.error(f"[WeCom] 文本消息发送异常: {e}")
            return False
    
    def send_daily_brief(self, 
                         script: str,
                         audio_path: str = None,
                         top_items: List[dict] = None) -> dict:
        """
        发送每日技术简报
        
        这是一个原子技能函数:
        - 输入: 播客讲稿、音频路径、Top 5 数据
        - 输出: 推送结果状态
        
        Args:
            script: 播客讲稿文本
            audio_path: 音频文件路径，如果为 None 则只发送图文
            top_items: Top 5 数据，用于生成图文卡片
            
        Returns:
            推送结果字典
        """
        results = {
            "success": False,
            "voice_sent": False,
            "card_sent": False,
            "errors": []
        }
        
        # 生成图文卡片描述
        today = datetime.now().strftime("%m月%d日")
        if top_items:
            item_list = "\n".join([
                f"{i+1}. {item.get('title', '')}"
                for i, item in enumerate(top_items[:5])
            ])
            description = f"今日精选 {len(top_items)} 条AI技术资讯：\n\n{item_list}\n\n点击收听完整语音播报 👆"
        else:
            description = f"今日AI技术资讯已送达，点击收听完整语音播报 👆"
        
        # 发送图文卡片
        card_success = self.send_text_card(
            title=f"🤖 AI每日技术简报 - {today}",
            description=description,
            url="https://github.com/trending/python"
        )
        results["card_sent"] = card_success
        
        if not card_success:
            results["errors"].append("图文卡片发送失败")
        
        # 发送语音消息
        if audio_path and Path(audio_path).exists():
            voice_success = self.send_voice(audio_path)
            results["voice_sent"] = voice_success
            
            if not voice_success:
                results["errors"].append("语音消息发送失败")
        else:
            logger.info("[WeCom] 无音频文件，跳过语音发送")
        
        results["success"] = results["card_sent"] or results["voice_sent"]
        
        return results


def send_notification(script: str,
                     audio_path: str = None,
                     top_items: List[dict] = None,
                     debug: bool = None) -> dict:
    """
    便捷函数: 发送每日简报通知
    
    使用示例:
        from writer import generate_podcast_script
        from audio import generate_audio
        from notifier import send_notification
        
        script = generate_podcast_script(top5, items)
        audio_path = generate_audio(script)
        result = send_notification(script, audio_path, top5)
    
    Args:
        script: 播客讲稿文本
        audio_path: 音频文件路径
        top_items: Top 5 数据
        debug: 是否调试模式，默认读取环境变量 DEBUG
        
    Returns:
        推送结果字典
    """
    # 检查 DEBUG 模式
    if debug is None:
        debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    if debug:
        logger.info("[Notifier] DEBUG 模式: 只打印消息，不实际发送")
        logger.info("=" * 50)
        logger.info("【图文卡片预览】")
        today = datetime.now().strftime("%m月%d日")
        logger.info(f"标题: 🤖 AI每日技术简报 - {today}")
        if top_items:
            logger.info("内容预览:")
            for i, item in enumerate(top_items[:5], 1):
                logger.info(f"  {i}. {item.get('title', '')}")
        logger.info("=" * 50)
        logger.info("【语音文件】")
        logger.info(f"路径: {audio_path or '无'}")
        logger.info("=" * 50)
        
        return {
            "success": True,
            "voice_sent": False,
            "card_sent": False,
            "debug": True,
            "message": "DEBUG 模式: 消息未实际发送"
        }
    
    # 正式发送
    notifier = WeComNotifier()
    return notifier.send_daily_brief(script, audio_path, top_items)


if __name__ == "__main__":
    # 测试运行
    from dotenv import load_dotenv
    load_dotenv()
    
    logger.info("=" * 50)
    logger.info("开始测试企业微信推送")
    logger.info("=" * 50)
    
    # 检查配置
    corp_id = os.getenv('CORP_ID')
    agent_id = os.getenv('AGENT_ID')
    secret = os.getenv('SECRET')
    
    if not all([corp_id, agent_id, secret]):
        logger.warning("[Test] 缺少企业微信配置，将使用 DEBUG 模式测试")
        os.environ['DEBUG'] = 'True'
    
    # 测试数据
    test_script = "大家好，欢迎收听AI每日技术简报。今天为大家介绍AutoGen和Swarm两个多智能体框架。"
    test_items = [
        {"title": "microsoft/autogen", "source": "github"},
        {"title": "openai/swarm", "source": "github"},
        {"title": "Tool Learning Survey", "source": "arxiv"}
    ]
    
    # 发送测试
    result = send_notification(test_script, None, test_items)
    
    logger.info("\n推送结果:")
    logger.info(json.dumps(result, ensure_ascii=False, indent=2))
