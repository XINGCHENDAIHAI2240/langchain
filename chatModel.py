from langchain.chat_models import init_chat_model
from langchain_community.chat_models.tongyi import ChatTongyi

# from langchain_community.llms.tongyi import Tongyi
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from customLLM import CustomLLM
from env_utils import (
    API_KEY,
    BASE_URL,
    MINIMAX_API_KEY,
    MINIMAX_BASE_URL,
    TONGYI_API_KEY,
)

# 示例：统一演示不同聊天模型的初始化与调用方式（不涉及业务逻辑）
# - ChatOpenAI：OpenAI 兼容接口（此处可对接 MiniMax 兼容端点）
# - ChatTongyi：阿里云通义千问对话模型
# - CustomLLM：项目自定义封装的模型类

# OpenAI 兼容接口，使用 langchain_openai 库
# temperature 越大，输出通常越发散；越小越稳定
llm_openai = ChatOpenAI(
    model="minmax-m2.5", temperature=0.5, api_key=API_KEY, base_url=BASE_URL
)

# 使用千问模型进行对话（需正确配置 TONGYI_API_KEY）
llm_tongyi = ChatTongyi(model="qwen-max", temperature=0.7, api_key=TONGYI_API_KEY)

# 使用自定义模型封装（便于统一扩展重试、日志、参数处理）
llm_custom = CustomLLM(
    model="MiniMax-M2.5", api_key=MINIMAX_API_KEY, base_url=MINIMAX_BASE_URL
)

# 内存速率限制器：控制请求速率，避免短时间内请求过多
# - requests_per_second=0.1: 平均 10 秒 1 次请求
# - check_every_n_seconds: 限流检查频率
# - max_bucket_size: 令牌桶容量，允许一定突发
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)

# 通过 init_chat_model 创建带限流能力的聊天模型实例
llm_rate_limiter = init_chat_model(
    model="MiniMax-M2.5",
    model_provider="openai",
    rate_limiter=rate_limiter,
    base_url=MINIMAX_BASE_URL,
    api_key=MINIMAX_API_KEY,
)


if __name__ == "__main__":
    # 流式调用：逐块输出模型回复内容
    res = llm_rate_limiter.stream("帮我讲个笑话吧")
    for item in res:
        print(item.content, end="")
