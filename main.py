"""
AI 每日技术新闻播报机器人 - 主入口

基于 Agentic Workflow 设计哲学:
- 管道优于自主 (Pipeline over Autonomy): 严格的线性管道
- 无状态 (Stateless): 每次运行独立
- 原子技能 (Atomic Skills): 各模块独立运行，错误隔离

工作流 Pipeline:
    数据抓取 (scraper) 
        ↓
    智能筛选 (processor) 
        ↓
    内容生成 (writer) 
        ↓
    语音合成 (audio) 
        ↓
    消息推送 (notifier)

使用方法:
    1. 安装依赖: pip install -r requirements.txt
    2. 配置环境: cp .env.example .env && 编辑 .env
    3. 运行一次: python main.py
    4. 定时运行: python main.py --schedule
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from loguru import logger

# 加载环境变量
load_dotenv()

# 导入原子技能模块
from scraper import fetch_all_sources, TechItem
from processor import select_top_items
from writer import generate_podcast_script
from audio import generate_audio
from notifier import send_notification


# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/daily_brief_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)


class DailyBriefPipeline:
    """
    每日技术简报工作流管道
    
    严格遵循线性管道设计，每个步骤的错误都被隔离，
    不会导致整个流程崩溃。
    """
    
    def __init__(self):
        self.debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
        self.results = {
            "start_time": None,
            "end_time": None,
            "steps": {},
            "success": False
        }
    
    def step_1_scrape(self) -> List[TechItem]:
        """
        步骤 1: 数据采集
        
        抓取 GitHub Trending、Hugging Face Papers、arXiv
        """
        logger.info("=" * 60)
        logger.info("[Step 1/5] 开始数据采集...")
        logger.info("=" * 60)
        
        try:
            items = fetch_all_sources()
            self.results["steps"]["scrape"] = {
                "status": "success",
                "count": len(items)
            }
            logger.info(f"[Step 1/5] 数据采集完成，共 {len(items)} 条")
            return items
        except Exception as e:
            logger.error(f"[Step 1/5] 数据采集失败: {e}")
            self.results["steps"]["scrape"] = {
                "status": "failed",
                "error": str(e)
            }
            return []
    
    def step_2_process(self, items: List[TechItem]) -> List[Dict[str, Any]]:
        """
        步骤 2: 智能筛选
        
        调用 GLM-4 对内容进行评分，选出 Top 5
        """
        logger.info("=" * 60)
        logger.info("[Step 2/5] 开始智能筛选...")
        logger.info("=" * 60)
        
        if not items:
            logger.warning("[Step 2/5] 无数据可筛选，跳过")
            self.results["steps"]["process"] = {
                "status": "skipped",
                "reason": "no_data"
            }
            return []
        
        try:
            top_items = select_top_items(items)
            self.results["steps"]["process"] = {
                "status": "success",
                "count": len(top_items)
            }
            logger.info(f"[Step 2/5] 智能筛选完成，选出 Top {len(top_items)}")
            
            # 打印筛选结果
            for i, item in enumerate(top_items, 1):
                logger.info(f"  {i}. [{item.get('score', 0):.1f}] {item.get('title', '')}")
            
            return top_items
        except Exception as e:
            logger.error(f"[Step 2/5] 智能筛选失败: {e}")
            self.results["steps"]["process"] = {
                "status": "failed",
                "error": str(e)
            }
            return []
    
    def step_3_write(self, top_items: List[Dict[str, Any]], 
                     all_items: List[TechItem]) -> str:
        """
        步骤 3: 内容生成
        
        将 Top 5 内容改写为中文播客讲稿
        """
        logger.info("=" * 60)
        logger.info("[Step 3/5] 开始内容生成...")
        logger.info("=" * 60)
        
        if not top_items:
            logger.warning("[Step 3/5] 无内容可生成，跳过")
            self.results["steps"]["write"] = {
                "status": "skipped",
                "reason": "no_data"
            }
            return ""
        
        try:
            script = generate_podcast_script(top_items, all_items)
            char_count = len(script.replace(' ', '').replace('\n', ''))
            
            self.results["steps"]["write"] = {
                "status": "success",
                "char_count": char_count
            }
            
            logger.info(f"[Step 3/5] 内容生成完成，共 {char_count} 字符")
            logger.info("-" * 60)
            logger.info("播客讲稿预览:")
            logger.info(script[:200] + "..." if len(script) > 200 else script)
            logger.info("-" * 60)
            
            return script
        except Exception as e:
            logger.error(f"[Step 3/5] 内容生成失败: {e}")
            self.results["steps"]["write"] = {
                "status": "failed",
                "error": str(e)
            }
            return ""
    
    def step_4_audio(self, script: str) -> Optional[str]:
        """
        步骤 4: 语音合成
        
        使用 edge-tts 将讲稿合成为语音
        """
        logger.info("=" * 60)
        logger.info("[Step 4/5] 开始语音合成...")
        logger.info("=" * 60)
        
        if not script:
            logger.warning("[Step 4/5] 无讲稿可合成，跳过")
            self.results["steps"]["audio"] = {
                "status": "skipped",
                "reason": "no_script"
            }
            return None
        
        # DEBUG 模式不跳过语音合成，只跳过消息推送
        # if self.debug_mode:
        #     logger.info("[Step 4/5] DEBUG 模式: 跳过语音合成")
        #     self.results["steps"]["audio"] = {
        #         "status": "skipped",
        #         "reason": "debug_mode"
        #     }
        #     return None
        
        try:
            audio_path = generate_audio(script)
            
            if audio_path:
                self.results["steps"]["audio"] = {
                    "status": "success",
                    "path": audio_path
                }
                logger.info(f"[Step 4/5] 语音合成完成: {audio_path}")
            else:
                self.results["steps"]["audio"] = {
                    "status": "failed",
                    "reason": "synthesis_failed"
                }
                logger.error("[Step 4/5] 语音合成失败")
            
            return audio_path
        except Exception as e:
            logger.error(f"[Step 4/5] 语音合成异常: {e}")
            self.results["steps"]["audio"] = {
                "status": "failed",
                "error": str(e)
            }
            return None
    
    def step_5_notify(self, script: str, audio_path: Optional[str],
                      top_items: List[Dict[str, Any]]) -> dict:
        """
        步骤 5: 消息推送
        
        通过企业微信发送图文卡片和语音消息
        """
        logger.info("=" * 60)
        logger.info("[Step 5/5] 开始消息推送...")
        logger.info("=" * 60)
        
        if not script:
            logger.warning("[Step 5/5] 无内容可推送，跳过")
            self.results["steps"]["notify"] = {
                "status": "skipped",
                "reason": "no_content"
            }
            return {"success": False, "reason": "no_content"}
        
        try:
            result = send_notification(
                script=script,
                audio_path=audio_path,
                top_items=top_items,
                debug=self.debug_mode
            )
            
            self.results["steps"]["notify"] = {
                "status": "success" if result.get("success") else "failed",
                "details": result
            }
            
            if result.get("success"):
                logger.info("[Step 5/5] 消息推送完成")
            else:
                logger.error(f"[Step 5/5] 消息推送失败: {result}")
            
            return result
        except Exception as e:
            logger.error(f"[Step 5/5] 消息推送异常: {e}")
            self.results["steps"]["notify"] = {
                "status": "failed",
                "error": str(e)
            }
            return {"success": False, "error": str(e)}
    
    def run(self) -> dict:
        """
        运行完整的工作流管道
        
        Returns:
            执行结果字典
        """
        self.results["start_time"] = datetime.now().isoformat()
        
        logger.info("\n" + "=" * 60)
        logger.info("🚀 AI 每日技术简报机器人启动")
        logger.info(f"🐛 DEBUG 模式: {self.debug_mode}")
        logger.info("=" * 60 + "\n")
        
        try:
            # Step 1: 数据采集
            items = self.step_1_scrape()
            
            # Step 2: 智能筛选
            top_items = self.step_2_process(items)
            
            # Step 3: 内容生成
            script = self.step_3_write(top_items, items)
            
            # Step 4: 语音合成
            audio_path = self.step_4_audio(script)
            
            # Step 5: 消息推送
            notify_result = self.step_5_notify(script, audio_path, top_items)
            
            # 判断整体成功
            self.results["success"] = (
                self.results["steps"].get("scrape", {}).get("status") == "success" and
                self.results["steps"].get("process", {}).get("status") == "success" and
                self.results["steps"].get("write", {}).get("status") == "success" and
                notify_result.get("success", False)
            )
            
        except Exception as e:
            logger.critical(f"工作流执行异常: {e}")
            self.results["success"] = False
            self.results["critical_error"] = str(e)
        
        finally:
            self.results["end_time"] = datetime.now().isoformat()
            
            # 计算执行时间
            if self.results["start_time"] and self.results["end_time"]:
                start = datetime.fromisoformat(self.results["start_time"])
                end = datetime.fromisoformat(self.results["end_time"])
                duration = (end - start).total_seconds()
                self.results["duration_seconds"] = duration
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ 工作流执行完成")
            logger.info(f"⏱️  执行时间: {self.results.get('duration_seconds', 0):.2f} 秒")
            logger.info(f"🎯 整体状态: {'成功' if self.results['success'] else '失败'}")
            logger.info("=" * 60 + "\n")
        
        return self.results


def run_once():
    """运行一次完整的工作流"""
    pipeline = DailyBriefPipeline()
    results = pipeline.run()
    return results


def run_schedule():
    """
    定时调度模式
    
    默认每天早上 9:00 执行一次
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    
    logger.info("🕐 启动定时调度模式")
    logger.info("⏰ 执行时间: 每天 09:00")
    
    scheduler = BlockingScheduler()
    
    # 每天早上 9:00 执行
    scheduler.add_job(
        run_once,
        trigger=CronTrigger(hour=9, minute=0),
        id='daily_brief',
        name='AI Daily Brief',
        replace_existing=True
    )
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("👋 收到中断信号，停止调度器")
        scheduler.shutdown()


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="AI 每日技术新闻播报机器人",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py              # 运行一次
    python main.py --schedule   # 定时模式 (每天 9:00)
    python main.py --debug      # 调试模式运行一次
        """
    )
    
    parser.add_argument(
        "--schedule", "-s",
        action="store_true",
        help="启动定时调度模式 (每天 09:00 执行)"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="强制开启 DEBUG 模式 (只打印，不发送)"
    )
    
    args = parser.parse_args()
    
    # 强制 DEBUG 模式
    if args.debug:
        os.environ['DEBUG'] = 'True'
        logger.info("🐛 强制开启 DEBUG 模式")
    
    # 运行模式
    if args.schedule:
        run_schedule()
    else:
        results = run_once()
        
        # 根据结果设置退出码
        if not results.get("success"):
            sys.exit(1)


if __name__ == "__main__":
    main()
