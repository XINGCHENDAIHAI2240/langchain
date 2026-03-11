from typing import Literal, TypedDict

from openai import OpenAI

from env_utils import API_KEY, BASE_URL


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class ClientOpenAI:
    def __init__(self, api_key: str, base_url: str):
        self.__client = OpenAI(api_key=api_key, base_url=base_url)
        self.__message: Message = []

    def __create_type_message(self) -> Message:
        exaples_type: Literal = [
            "新闻报道",
            "财务报道",
            "公司公告",
            "分析师报告",
        ]
        exaples_data: dict = {
            "新闻报道": "今日某科技公司宣布推出新一代AI产品，"
            "市场反应热烈，股价上涨5%。",
            "财务报道": "该公司季度营收达到100亿元，同比增长20%，超出市场预期。",
            "公司公告": "董事会宣布将于下月召开年度股东大会，讨论分红方案。",
            "分析报告": "分析师上调目标价至150元，维持买入评级，看好公司长期发展。",
        }
        system_prompt = (
            f"你是一个金融专家，将文本分类为:{exaples_type}," "不清楚的分类为不清楚类别"
        )
        self.__message = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]

        # 将exaples_data 数据添加到 message
        for key, value in exaples_data.items():
            self.__message.append({"role": "user", "content": value})
            self.__message.append({"role": "assistant", "content": key})

        return self.__message

    def question_type(self, qs: str, model: str = "minimax-m2.5") -> str:
        self.__create_type_message()
        self.__message.append(
            {"role": "user", "content": f"按照上述提示，回答这段文本的分类类别:{qs}"}
        )
        res = self.__client.chat.completions.create(
            model=model,
            messages=self.__message,
        )
        return res.choices[0].message.content

    def __create_schema_message(self) -> Message:
        _schema = ["日期", "股票名称", "开盘价", "收盘价", "成交量"]
        _exaples_data = [
            {
                "content": (
                    "2024年1月15日，阿里巴巴股票开盘价为124.20元，"
                    "收盘价为125.50元，当日成交1500万股。"
                ),
                "answers": {
                    "日期": "2024年1月15日",
                    "股票名称": "阿里巴巴",
                    "开盘价": "124.20元",
                    "收盘价": "125.50元",
                    "成交量": "1500万股",
                },
            },
            {
                "content": (
                    "腾讯控股在2024年2月20日的开盘价为380.00港元，"
                    "收盘价为385.50港元，成交量达到2000万股。"
                ),
                "answers": {
                    "日期": "2024年2月20日",
                    "股票名称": "腾讯控股",
                    "开盘价": "380.00港元",
                    "收盘价": "385.50港元",
                    "成交量": "2000万股",
                },
            },
            {
                "content": (
                    "2024年3月10日，苹果公司股票开盘价为174.50美元，"
                    "收盘价为175.30美元，当日成交800万股。"
                ),
                "answers": {
                    "日期": "2024年3月10日",
                    "股票名称": "苹果公司",
                    "开盘价": "174.50美元",
                    "收盘价": "175.30美元",
                    "成交量": "800万股",
                },
            },
        ]
        # TODO: 实现完整方法逻辑
        pass


if __name__ == "__main__":
    # 测试文本 - 用于分类测试
    # test_texts: list = [
    #     # 新闻报道
    #     "华为发布最新款Mate手机，引领5G时代创新潮流",
    #     # 财务报道
    #     "苹果公司2024年Q3净利润同比增长15%，达到230亿美元",
    #     # 公司公告
    #     "万科地产公告：将延期举办2024年度投资者交流会",
    #     # 分析师报告
    #     "中金公司发布研报：维持宁德时代买入评级，目标价上调至600元",
    #     # 不清楚类别
    #     "今天天气真不错，适合出去游玩",
    # ]

    client = ClientOpenAI(api_key=API_KEY, base_url=BASE_URL)

    # for text in test_texts:
    #     result = client.question_type(text)
    #     print(f"文本: {text[:30]}...")
    #     print(f"分类: {result}")
    #     print("-" * 50)
