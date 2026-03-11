from typing import Literal, TypedDict

from openai import OpenAI


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class ClientOpenAI:
    def __init__(self, api_key: str, base_url: str):
        self.__client = OpenAI(api_key=api_key, base_url=base_url)
        self.__exaples_type: Literal = [
            "新闻报道",
            "财务报道",
            "公司公告",
            "分析师报告",
        ]
        self.__exaples_data: dict = {
            "新闻报道": "今日某科技公司宣布推出新一代AI产品，"
            "市场反应热烈，股价上涨5%。",
            "财务报道": "该公司季度营收达到100亿元，同比增长20%，超出市场预期。",
            "公司公告": "董事会宣布将于下月召开年度股东大会，讨论分红方案。",
            "分析师报告": "分析师上调目标价至150元，维持买入评级，看好公司长期发展。",
        }
        self.__question: list = [
            "该公司最近有什么重大新闻？",
            "请总结一下本季度的财务表现。",
            "什么时候召开股东大会？",
            "分析师对公司有什么评价？",
            "公司发布了哪些公告？",
        ]
