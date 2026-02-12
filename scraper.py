"""
数据采集技能 (scraper.py)

基于 Agentic Workflow 设计哲学:
- 原子技能: 纯函数封装，无状态，独立运行
- 强健错误处理: 单个源失败不崩溃整个流程

数据源:
1. GitHub Trending (Python, AI/ML 标签)
2. Hugging Face Daily Papers
3. arXiv (cs.CL, cs.AI)
4. 中国AI权威数据源
"""

import os
import re
import json
import time
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from loguru import logger


@dataclass
class TechItem:
    """技术新闻条目数据结构"""
    source: str           # 来源: github, huggingface, arxiv
    title: str            # 标题
    url: str              # 链接
    description: str      # 描述/摘要
    author: Optional[str] = None  # 作者
    stars: Optional[int] = None   # GitHub stars
    language: Optional[str] = None  # 编程语言
    published: Optional[str] = None  # 发布日期


class BaseScraper:
    """抓取器基类 - 定义统一接口"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        # 设置代理(如果环境变量中有)
        if os.getenv('HTTP_PROXY'):
            self.session.proxies = {
                'http': os.getenv('HTTP_PROXY'),
                'https': os.getenv('HTTPS_PROXY', os.getenv('HTTP_PROXY'))
            }
    
    def fetch(self) -> List[TechItem]:
        """子类必须实现的抓取方法"""
        raise NotImplementedError


class ChinaAIScraper(BaseScraper):
    """中国AI权威数据源抓取器"""
    
    def fetch(self) -> List[TechItem]:
        """抓取国内权威AI数据源"""
        items = []
        try:
            # 1. 百度开发者社区
            items.extend(self._fetch_baidu_developer())
            
            # 2. 阿里云开发者社区
            items.extend(self._fetch_aliyun_developer())
            
            # 3. 腾讯云开发者社区
            items.extend(self._fetch_tencent_cloud())
            
            # 4. 华为云开发者社区
            items.extend(self._fetch_huawei_cloud())
            
            # 5. CSDN - 技术社区
            items.extend(self._fetch_csdn())
            
            # 6. 机器之心 - AI专业媒体
            items.extend(self._fetch_jiqizhixin())
            
            # 7. 雷锋网 - 科技媒体
            items.extend(self._fetch_leiphone())
            
            logger.info(f"[ChinaAI] 成功抓取 {len(items)} 条国内AI资讯")
            
        except Exception as e:
            logger.error(f"[ChinaAI] 抓取失败: {e}")
        
        return items
    
    def _fetch_jiqizhixin(self) -> List[TechItem]:
        """抓取机器之心 - AI专业媒体"""
        items = []
        try:
            url = "https://www.jiqizhixin.com/"
            logger.info(f"[ChinaAI] 开始抓取机器之心: {url}")
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找文章链接
            links = soup.find_all('a', href=True)
            
            seen = set()
            for link in links:
                try:
                    href = link.get('href', '')
                    if not href or href in seen:
                        continue
                    
                    # 筛选AI相关链接
                    if '/articles/' in href or 'ai' in href.lower():
                        seen.add(href)
                        article_url = href if href.startswith('http') else f"https://www.jiqizhixin.com{href}"
                        
                        # 提取标题
                        title = link.get_text(strip=True) or "Unknown"
                        if len(title) < 8:  # 过滤太短的标题
                            continue
                        
                        items.append(TechItem(
                            source='jiqizhixin',
                            title=title,
                            url=article_url,
                            description="机器之心AI技术动态"
                        ))
                        
                        if len(items) >= 2:  # 限制数量
                            break
                            
                except Exception as e:
                    logger.warning(f"[ChinaAI] 解析机器之心链接失败: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"[ChinaAI] 抓取机器之心失败: {e}")
        
        return items
    
    def _fetch_leiphone(self) -> List[TechItem]:
        """抓取雷锋网 - 科技媒体"""
        items = []
        try:
            url = "https://www.leiphone.com/"
            logger.info(f"[ChinaAI] 开始抓取雷锋网: {url}")
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找文章链接
            links = soup.find_all('a', href=True)
            
            seen = set()
            for link in links:
                try:
                    href = link.get('href', '')
                    if not href or href in seen:
                        continue
                    
                    # 筛选AI相关链接
                    if '/category/ai/' in href or 'ai' in href.lower():
                        seen.add(href)
                        article_url = href if href.startswith('http') else f"https://www.leiphone.com{href}"
                        
                        # 提取标题
                        title = link.get_text(strip=True) or "Unknown"
                        if len(title) < 8:  # 过滤太短的标题
                            continue
                        
                        items.append(TechItem(
                            source='leiphone',
                            title=title,
                            url=article_url,
                            description="雷锋网AI技术动态"
                        ))
                        
                        if len(items) >= 2:  # 限制数量
                            break
                            
                except Exception as e:
                    logger.warning(f"[ChinaAI] 解析雷锋网链接失败: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"[ChinaAI] 抓取雷锋网失败: {e}")
        
        return items
    
    def _fetch_baidu_developer(self) -> List[TechItem]:
        """抓取百度开发者社区"""
        items = []
        try:
            # 尝试多个URL
            urls = [
                "https://ai.baidu.com/tech/",
                "https://ai.baidu.com/",
                "https://www.baidu.com/s?wd=AI%E6%8A%80%E6%9C%AF"
            ]
            
            for url in urls:
                try:
                    logger.info(f"[ChinaAI] 尝试抓取百度开发者社区: {url}")
                    
                    # 添加更完整的请求头
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1'
                    }
                    
                    response = self.session.get(url, timeout=10, headers=headers)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 查找文章链接
                    links = soup.find_all('a', href=True)
                    
                    seen = set()
                    for link in links:
                        try:
                            href = link.get('href', '')
                            if not href or href in seen:
                                continue
                            
                            # 筛选AI相关链接
                            if '/tech/' in href or '/blog/' in href or '/article/' in href:
                                seen.add(href)
                                article_url = f"https://ai.baidu.com{href}" if href.startswith('/') else href
                                
                                # 提取标题
                                title = link.get_text(strip=True) or "Unknown"
                                if len(title) < 8:  # 过滤太短的标题
                                    continue
                                
                                items.append(TechItem(
                                    source='baidu_dev',
                                    title=title,
                                    url=article_url,
                                    description="百度AI技术动态"
                                ))
                                
                                if len(items) >= 2:  # 限制数量
                                    break
                                    
                        except Exception as e:
                            logger.warning(f"[ChinaAI] 解析百度开发者社区链接失败: {e}")
                            continue
                    
                    if items:  # 如果成功获取到数据，就不再尝试其他URL
                        break
                        
                except Exception as e:
                    logger.warning(f"[ChinaAI] 抓取百度开发者社区URL失败: {url}, 错误: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"[ChinaAI] 抓取百度开发者社区失败: {e}")
        
        return items
    
    def _fetch_aliyun_developer(self) -> List[TechItem]:
        """抓取阿里云开发者社区"""
        items = []
        try:
            urls = [
                "https://developer.aliyun.com/ai",
                "https://developer.aliyun.com/"
            ]
            
            for url in urls:
                try:
                    logger.info(f"[ChinaAI] 开始抓取阿里云开发者社区: {url}")
                    
                    response = self.session.get(url, timeout=15)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 查找文章链接
                    links = soup.find_all('a', href=True)
                    
                    seen = set()
                    for link in links:
                        try:
                            href = link.get('href', '')
                            if not href or href in seen:
                                continue
                            
                            # 筛选AI相关链接
                            if '/article/' in href or '/blog/' in href or 'ai' in href.lower():
                                seen.add(href)
                                article_url = href if href.startswith('http') else f"https://developer.aliyun.com{href}"
                                
                                # 提取标题
                                title = link.get_text(strip=True) or "Unknown"
                                if len(title) < 8:  # 过滤太短的标题
                                    continue
                                
                                items.append(TechItem(
                                    source='aliyun_dev',
                                    title=title,
                                    url=article_url,
                                    description="阿里云AI技术动态"
                                ))
                                
                                if len(items) >= 2:  # 限制数量
                                    break
                                    
                        except Exception as e:
                            logger.warning(f"[ChinaAI] 解析阿里云开发者社区失败: {e}")
                            continue
                    
                    if items:  # 如果成功获取到数据，就不再尝试其他URL
                        break
                        
                except Exception as e:
                    logger.warning(f"[ChinaAI] 抓取阿里云开发者社区失败: {url}, 错误: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"[ChinaAI] 抓取阿里云开发者社区失败: {e}")
        
        return items
    
    def _fetch_tencent_cloud(self) -> List[TechItem]:
        """抓取腾讯云开发者社区"""
        items = []
        try:
            url = "https://cloud.tencent.com/developer/column/10755"
            logger.info(f"[ChinaAI] 开始抓取腾讯云开发者社区: {url}")
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找文章链接
            links = soup.find_all('a', href=True)
            
            seen = set()
            for link in links:
                try:
                    href = link.get('href', '')
                    if not href or href in seen:
                        continue
                    
                    # 筛选AI相关链接
                    if '/developer/article/' in href:
                        seen.add(href)
                        article_url = href if href.startswith('http') else f"https://cloud.tencent.com{href}"
                        
                        # 提取标题
                        title = link.get_text(strip=True) or "Unknown"
                        if len(title) < 8:  # 过滤太短的标题
                            continue
                        
                        items.append(TechItem(
                            source='tencent_cloud',
                            title=title,
                            url=article_url,
                            description="腾讯云AI技术动态"
                        ))
                        
                        if len(items) >= 2:  # 限制数量
                            break
                            
                except Exception as e:
                    logger.warning(f"[ChinaAI] 解析腾讯云开发者社区失败: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"[ChinaAI] 抓取腾讯云开发者社区失败: {e}")
        
        return items
    
    def _fetch_huawei_cloud(self) -> List[TechItem]:
        """抓取华为云开发者社区"""
        items = []
        try:
            # 尝试多个URL
            urls = [
                "https://www.huaweicloud.com/intl/zh-cn/products/ai.html",
                "https://www.huaweicloud.com/product/ai.html",
                "https://www.huaweicloud.com/zh-cn/product/ai.html",
                "https://www.huaweicloud.com/"
            ]
            
            for url in urls:
                try:
                    logger.info(f"[ChinaAI] 尝试抓取华为云开发者社区: {url}")
                    
                    # 添加更完整的请求头
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1'
                    }
                    
                    response = self.session.get(url, timeout=10, headers=headers)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 查找文章链接
                    links = soup.find_all('a', href=True)
                    
                    seen = set()
                    for link in links:
                        try:
                            href = link.get('href', '')
                            if not href or href in seen:
                                continue
                            
                            # 筛选AI相关链接
                            if '/blog/' in href or '/article/' in href or '/product/' in href:
                                seen.add(href)
                                article_url = href if href.startswith('http') else f"https://www.huaweicloud.com{href}"
                                
                                # 提取标题
                                title = link.get_text(strip=True) or "Unknown"
                                if len(title) < 8:  # 过滤太短的标题
                                    continue
                                
                                items.append(TechItem(
                                    source='huawei_cloud',
                                    title=title,
                                    url=article_url,
                                    description="华为云AI技术动态"
                                ))
                                
                                if len(items) >= 2:  # 限制数量
                                    break
                                    
                        except Exception as e:
                            logger.warning(f"[ChinaAI] 解析华为云开发者社区链接失败: {e}")
                            continue
                    
                    if items:  # 如果成功获取到数据，就不再尝试其他URL
                        break
                        
                except Exception as e:
                    logger.warning(f"[ChinaAI] 抓取华为云开发者社区URL失败: {url}, 错误: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"[ChinaAI] 抓取华为云开发者社区失败: {e}")
        
        return items
    
    def _fetch_csdn(self) -> List[TechItem]:
        """抓取CSDN技术社区"""
        items = []
        try:
            # 尝试多个URL，移除无效的topic/ai URL
            urls = [
                "https://www.csdn.net/nav/ai",
                "https://www.csdn.net/",
                "https://blog.csdn.net/"
            ]
            
            for url in urls:
                try:
                    logger.info(f"[ChinaAI] 尝试抓取CSDN技术社区: {url}")
                    
                    # 添加更完整的请求头
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1'
                    }
                    
                    response = self.session.get(url, timeout=10, headers=headers)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 查找文章链接
                    links = soup.find_all('a', href=True)
                    
                    seen = set()
                    for link in links:
                        try:
                            href = link.get('href', '')
                            if not href or href in seen:
                                continue
                            
                            # 筛选AI相关链接
                            if '/article/details/' in href or '/nav/ai' in href or 'ai' in href.lower():
                                seen.add(href)
                                # 确保URL格式正确
                                if href.startswith('http'):
                                    article_url = href
                                elif href.startswith('/'):
                                    article_url = f"https://www.csdn.net{href}"
                                else:
                                    article_url = f"https://www.csdn.net/{href}"
                                
                                # 提取标题
                                title = link.get_text(strip=True) or "Unknown"
                                if len(title) < 8:  # 过滤太短的标题
                                    continue
                                
                                items.append(TechItem(
                                    source='csdn',
                                    title=title,
                                    url=article_url,
                                    description="CSDN AI技术文章"
                                ))
                                
                                if len(items) >= 2:  # 限制数量
                                    break
                                    
                        except Exception as e:
                            logger.warning(f"[ChinaAI] 解析CSDN链接失败: {e}")
                            continue
                    
                    if items:  # 如果成功获取到数据，就不再尝试其他URL
                        break
                        
                except Exception as e:
                    logger.warning(f"[ChinaAI] 抓取CSDN URL失败: {url}, 错误: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"[ChinaAI] 抓取CSDN失败: {e}")
        
        return items


class GitHubTrendingScraper(BaseScraper):
    """GitHub Trending 抓取器"""
    
    def fetch(self) -> List[TechItem]:
        """抓取 GitHub Trending Python 项目"""
        items = []
        try:
            # 抓取 Python 语言的 trending
            url = "https://github.com/trending/python?since=daily"
            logger.info(f"[GitHub] 开始抓取: {url}")
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 解析 trending 项目
            articles = soup.find_all('article', class_='Box-row')
            
            for article in articles[:10]:  # 取前10个
                try:
                    # 提取项目名
                    h2 = article.find('h2')
                    if not h2:
                        continue
                    
                    repo_link = h2.find('a')
                    if not repo_link:
                        continue
                    
                    repo_path = repo_link.get('href', '').strip('/')
                    repo_url = f"https://github.com/{repo_path}"
                    
                    # 提取描述
                    desc_tag = article.find('p', class_='col-9')
                    description = desc_tag.text.strip() if desc_tag else ""
                    
                    # 提取 stars
                    stars_tag = article.find('a', href=lambda x: x and 'stargazers' in x)
                    stars = 0
                    if stars_tag:
                        stars_text = stars_tag.text.strip().replace(',', '')
                        try:
                            stars = int(re.search(r'(\d+)', stars_text).group(1))
                        except:
                            pass
                    
                    # 聚焦 AI 编程、大模型、数据标注、AI 企业落地，增加更多中国相关内容
                    tech_keywords = [
                        # AI/ML 基础
                        'ai', 'ml', 'machine learning', 'deep learning', 'neural network',
                        # 大模型
                        'llm', 'gpt', 'large language model', 'transformer', 'foundation model',
                        'model release', 'new model', 'model launch', 'announce', 'release', 'launch',
                        # 编程和开发
                        'python', 'coding', 'programming', 'developer', 'software engineering',
                        'code generation', 'ai programming', 'developer tools',
                        # 框架和库
                        'pytorch', 'tensorflow', 'keras', 'numpy', 'pandas', 'scikit-learn',
                        # 大模型工具
                        'langchain', 'llama-index', 'openai', 'anthropic', 'huggingface',
                        # 数据标注
                        'data annotation', 'data labeling', 'dataset', 'data collection',
                        'labeling', 'annotation tool', 'data quality',
                        # AI 企业落地
                        'enterprise ai', 'business ai', 'industry ai', 'ai adoption',
                        'implementation', 'deployment', 'production', 'scaling', 'roi',
                        # 应用领域
                        'nlp', 'natural language processing', 'text generation', 'chatbot',
                        # 工具和技术
                        'rag', 'retrieval', 'embedding', 'fine-tuning', 'inference', 'deployment',
                        # 开发工具
                        'api', 'sdk', 'library', 'framework', 'tool', 'cli',
                        # 热门技术
                        'generative ai', 'ai assistant', 'prompt engineering',
                        # 知名大模型
                        'gemini', 'claude', 'mistral', 'llama', 'bloom', 'falcon', 'gpt-5', 'gemini 2.0', 'claude 3',
                        # GLM系列
                        'glm5', 'glm-5', 'GLM5', 'GLM-5', 'glm', 'GLM',
                        # 中国相关 - 国家和政策
                        'china', 'chinese', '中文', '中国', '国产化', '国内',
                        '中国AI', '中国人工智能', '国产AI', '国家战略', '新基建', '数字中国',
                        'AI 2.0', '人工智能 2.0', '国家实验室', '科技创新', '科技自强',
                        # 中国公司 - 科技巨头
                        'baidu', '字节跳动', 'tencent', 'alibaba', '华为', '小米',
                        '百度', 'bytedance', '腾讯', '阿里', 'huawei', 'xiaomi',
                        # 中国公司 - AI 专业公司
                        '智谱', '讯飞', '商汤', '旷视', '云从', '第四范式',
                        'zhupu', 'iflytek', 'sensetime', 'megvii', 'cloudwalk', '4paradigm',
                        '达闼', '零一万物', '百川智能', '深鉴科技', '地平线', '寒武纪',
                        '大模型', 'ai', '人工智能', '机器学习', '深度学习',
                        # 中国大模型
                        'GLM', 'ERNIE', '文心一言', '讯飞星火', '通义千问', '豆包',
                        'gemini', 'gpt', 'llm', 'large language model',
                        '智谱GLM', '百度ERNIE', 'ERNIE Bot', '文心', '星火', '通义', '豆包',
                        # 中国AI 应用场景
                        '智能制造', '智慧医疗', '智慧城市', '智能交通', '金融科技', '教育科技',
                        '工业AI', '医疗AI', '城市AI', '交通AI', '金融AI', '教育AI',
                        # 中国AI 技术
                        '多模态', '计算机视觉', '语音识别', '自然语言处理', '知识图谱',
                        'multimodal', 'computer vision', 'speech recognition', 'nlp',
                        # 中国AI 生态
                        'AI 生态', '人工智能生态', 'AI 产业链', '人工智能产业链',
                        '开源社区', 'open source', 'developer community', '开发者社区'
                    ]
                    
                    text_to_check = (repo_path + " " + description).lower()
                    is_tech_related = any(kw in text_to_check for kw in tech_keywords)
                    
                    if is_tech_related or stars > 100:
                        items.append(TechItem(
                            source='github',
                            title=repo_path,
                            url=repo_url,
                            description=description[:200],
                            stars=stars,
                            language='Python'
                        ))
                        
                except Exception as e:
                    logger.warning(f"[GitHub] 解析单个项目失败: {e}")
                    continue
            
            logger.info(f"[GitHub] 成功抓取 {len(items)} 个项目")
            
        except Exception as e:
            logger.error(f"[GitHub] 抓取失败: {e}")
        
        return items


class HuggingFaceScraper(BaseScraper):
    """Hugging Face Daily Papers 抓取器"""
    
    def fetch(self) -> List[TechItem]:
        """抓取 Hugging Face 每日论文"""
        items = []
        try:
            # 尝试多个Hugging Face相关URL
            urls = [
                "https://huggingface.co/papers",
                "https://huggingface.co/models?pipeline_tag=text-generation&sort=trending"
            ]
            
            for url in urls:
                try:
                    logger.info(f"[HuggingFace] 开始抓取: {url}")
                    
                    response = self.session.get(url, timeout=15)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 尝试多种方式查找链接
                    paper_links = []
                    
                    # 方式1: 原始正则匹配论文
                    paper_links.extend(soup.find_all('a', href=re.compile(r'/papers/\d+\.\d+')))
                    
                    # 方式2: 查找包含papers的链接
                    paper_links.extend(soup.find_all('a', href=lambda x: x and '/papers/' in x))
                    
                    # 方式3: 查找包含models的链接（用于models页面）
                    paper_links.extend(soup.find_all('a', href=lambda x: x and '/models/' in x))
                    
                    # 方式4: 查找卡片元素内的链接
                    paper_cards = soup.find_all(class_=re.compile(r'paper-card|card|model-card'))
                    for card in paper_cards:
                        link = card.find('a', href=lambda x: x and ('/papers/' in x or '/models/' in x))
                        if link:
                            paper_links.append(link)
                    
                    seen = set()
                    for link in paper_links[:15]:  # 取前15篇以提高成功率
                        try:
                            href = link.get('href', '')
                            if not href or href in seen:
                                continue
                            seen.add(href)
                            
                            paper_url = f"https://huggingface.co{href}" if href.startswith('/') else href
                            
                            # 提取标题
                            title_tag = link.find('h3') or link.find('h4') or link.find('h2') or link.find('h1') or link
                            title = title_tag.get_text(strip=True) if title_tag else "Unknown"
                            if len(title) < 10:  # 过滤太短的标题
                                continue
                            
                            # 提取摘要/描述
                            desc_tag = link.find_next('p')
                            if not desc_tag:
                                # 尝试从卡片中查找摘要
                                card = link.find_parent(class_=re.compile(r'paper-card|card|model-card'))
                                if card:
                                    # 尝试查找不同类型的描述元素
                                    desc_tag = card.find('p') or card.find(class_=re.compile(r'description|summary'))
                            description = desc_tag.get_text(strip=True)[:300] if desc_tag else ""
                            
                            # 聚焦大模型相关内容
                            llm_keywords = [
                                # 大模型基础术语
                                'llm', 'gpt', 'large language model', 'transformer', 'foundation model',
                                'model release', 'new model', 'model launch', 'announce', 'release', 'launch',
                                # 模型类型
                                'text generation', 'chat model', 'multimodal', 'vision-language',
                                # 知名模型
                                'gemini', 'claude', 'mistral', 'llama', 'bloom', 'falcon',
                                # 中国大模型
                                'glm', 'ernie', '文心一言', '讯飞星火', '通义千问', '豆包',
                                # 技术关键词
                                'fine-tuning', 'inference', 'deployment', 'scaling'
                            ]
                            
                            # 检查是否包含大模型相关内容
                            text_to_check = (title + " " + description).lower()
                            is_llm_related = any(kw in text_to_check for kw in llm_keywords)
                            
                            if is_llm_related:
                                items.append(TechItem(
                                    source='huggingface',
                                    title=title,
                                    url=paper_url,
                                    description=description
                                ))
                            
                            # 限制数量
                            if len(items) >= 6:
                                break
                            
                        except Exception as e:
                            logger.warning(f"[HuggingFace] 解析单篇内容失败: {e}")
                            continue
                    
                    if items:
                        break
                        
                except Exception as e:
                    logger.warning(f"[HuggingFace] 抓取URL失败: {url}, 错误: {e}")
                    continue
            
            # 如果仍然没有抓取到内容，使用备用策略
            if not items:
                logger.warning("[HuggingFace] 未抓取到内容，使用备用策略")
                # 添加大模型相关的备用项
                items.extend([
                    TechItem(
                        source='huggingface',
                        title='Hugging Face 热门大模型',
                        url='https://huggingface.co/models',
                        description='Hugging Face上最新发布和热门的大模型'
                    ),
                    TechItem(
                        source='huggingface',
                        title='大模型技术论文',
                        url='https://huggingface.co/papers',
                        description='最新的大模型研究论文和技术进展'
                    )
                ])
            
            logger.info(f"[HuggingFace] 成功抓取 {len(items)} 条大模型相关内容")
            
        except Exception as e:
            logger.error(f"[HuggingFace] 抓取失败: {e}")
            # 出错时添加备用项
            items.append(TechItem(
                source='huggingface',
                title='大模型研究动态',
                url='https://huggingface.co/models',
                description='Hugging Face最新大模型发布和研究成果'
            ))
        
        return items


class ArXivScraper(BaseScraper):
    """arXiv 论文抓取器"""
    
    def fetch(self) -> List[TechItem]:
        """抓取 arXiv cs.CL 和 cs.AI 最新论文"""
        items = []
        try:
            # 使用 arXiv API 获取最近提交的论文
            categories = ['cs.CL', 'cs.AI', 'cs.LG']
            
            for category in categories:
                try:
                    url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&max_results=5"
                    logger.info(f"[arXiv] 开始抓取 {category}")
                    
                    response = self.session.get(url, timeout=15)
                    response.raise_for_status()
                    
                    # 解析 XML
                    soup = BeautifulSoup(response.text, 'xml')
                    entries = soup.find_all('entry')
                    
                    for entry in entries:
                        try:
                            title = entry.find('title')
                            title = title.text.strip().replace('\n', ' ') if title else "Unknown"
                            
                            link = entry.find('id')
                            paper_url = link.text.strip() if link else ""
                            
                            summary = entry.find('summary')
                            description = summary.text.strip()[:300] if summary else ""
                            
                            author_tags = entry.find_all('author')
                            authors = [a.find('name').text for a in author_tags if a.find('name')]
                            author = authors[0] if authors else None
                            
                            published = entry.find('published')
                            published_date = published.text[:10] if published else None
                            
                            # 聚焦大模型相关内容，包括中国和国际大模型
                            llm_keywords = [
                                # 大模型基础术语
                                'llm', 'gpt', 'large language model', 'transformer', 'foundation model',
                                'model release', 'new model', 'model launch', 'announce', 'release', 'launch',
                                # 模型类型
                                'text generation', 'chat model', 'multimodal', 'vision-language',
                                # 知名模型
                                'gemini', 'claude', 'mistral', 'llama', 'bloom', 'falcon',
                                # 中国大模型
                                'glm', 'ernie', '文心一言', '讯飞星火', '通义千问', '豆包',
                                # 技术关键词
                                'fine-tuning', 'inference', 'deployment', 'scaling', 'rag',
                                # 中国相关
                                'china', 'chinese', '中文', '中国', 'baidu', 'tencent', 'alibaba',
                                # 基础AI术语
                                'ai', 'machine learning', 'deep learning', 'neural network',
                                'nlp', 'natural language processing', 'computer vision'
                            ]
                            
                            # 检查是否包含大模型相关内容
                            text_to_check = (title + " " + description).lower()
                            is_llm_related = any(kw in text_to_check for kw in llm_keywords)
                            
                            # 检查发布时间，只抓取当天的内容
                            is_recent = False
                            if published_date:
                                try:
                                    from datetime import datetime, timedelta
                                    pub_date = datetime.strptime(published_date, '%Y-%m-%d')
                                    today = datetime.now().date()
                                    pub_date_only = pub_date.date()
                                    is_recent = pub_date_only == today
                                except:
                                    is_recent = False  # 解析失败时默认认为不是当天
                            
                            if is_llm_related or is_recent:
                                items.append(TechItem(
                                    source='arxiv',
                                    title=title,
                                    url=paper_url,
                                    description=description,
                                    author=author,
                                    published=published_date
                                ))
                            
                        except Exception as e:
                            logger.warning(f"[arXiv] 解析单篇论文失败: {e}")
                            continue
                    
                    time.sleep(1)  # 礼貌延迟
                    
                except Exception as e:
                    logger.error(f"[arXiv] 抓取 {category} 失败: {e}")
                    continue
            
            logger.info(f"[arXiv] 成功抓取 {len(items)} 篇论文")
            
        except Exception as e:
            logger.error(f"[arXiv] 抓取失败: {e}")
        
        return items


def fetch_all_sources() -> List[TechItem]:
    """
    抓取所有数据源
    
    这是一个原子技能函数:
    - 无状态: 每次调用独立运行
    - 错误隔离: 单个源失败不影响其他源
    - 返回: 所有数据源的 TechItem 列表
    """
    all_items = []
    
    scrapers = [
        GitHubTrendingScraper(),
        HuggingFaceScraper(),
        ArXivScraper(),
        ChinaAIScraper(),  # 添加中国权威AI数据源
    ]
    
    for scraper in scrapers:
        try:
            items = scraper.fetch()
            all_items.extend(items)
        except Exception as e:
            logger.error(f"抓取器 {scraper.__class__.__name__} 异常: {e}")
            continue
    
    logger.info(f"[Scraper] 总计抓取 {len(all_items)} 条数据")
    return all_items


if __name__ == "__main__":
    # 测试运行
    from dotenv import load_dotenv
    load_dotenv()
    
    logger.info("=" * 50)
    logger.info("开始测试数据抓取")
    logger.info("=" * 50)
    
    results = fetch_all_sources()
    
    for item in results[:5]:
        logger.info(f"\n来源: {item.source}")
        logger.info(f"标题: {item.title}")
        logger.info(f"链接: {item.url}")
        logger.info(f"描述: {item.description[:100]}...")
        logger.info("-" * 50)