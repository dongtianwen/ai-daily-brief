## 实施计划

### 阶段一：项目初始化
1. **requirements.txt** - 依赖清单 (zhipuai, edge-tts, crawl4ai, python-dotenv, apscheduler等)
2. **.env.example** - 配置模板 (ZHIPUAI_API_KEY, 企业微信配置, DEBUG模式)

### 阶段二：原子技能开发
3. **scraper.py** - 数据采集技能 (GitHub Trending, Hugging Face Daily Papers, arXiv)
4. **processor.py** - GLM-4 智能筛选 Agent (评分0-10, Agent关键词权重×1.5, JSON输出)
5. **writer.py** - 内容生成 Agent (中文播客讲稿, 600-900字符, 术语发音矫正)
6. **audio.py** - TTS技能 (edge-tts生成MP3)
7. **notifier.py** - 企业微信推送技能 (语音+图文卡片)

### 阶段三：工作流编排
8. **main.py** - 线性管道编排 (数据抓取→智能筛选→反思改写→语音合成→消息推送)

## 核心设计原则
- **管道优于自主**: 严格线性管道，非自主Agent
- **无状态**: 每日运行独立，无记忆模块
- **原子技能**: 纯Python函数，不引入LangChain等复杂工具箱
- **强健错误处理**: 单个源失败不崩溃整个流程