from typing import Any, Iterator, Literal

from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from customLLM import CustomLLM
from env_utils import (
    API_KEY,
    BASE_URL,
    MINIMAX_API_KEY,
    MINIMAX_BASE_URL,
    TONGYI_API_KEY,
    TONGYI_BASE_URL,
)

# 支持的模型类型。
# 使用 Literal 可以让编辑器在传参时给出更明确的提示。
ModelType = Literal["openai", "tongyi", "custom", "rate_limited"]


class ChatModelService:
    """统一封装项目中的聊天模型初始化与调用逻辑。"""

    def __init__(self) -> None:
        # 使用字典缓存模型实例，避免重复初始化。
        self._models: dict[str, Any] = {}

    def get_model(self, model_type: ModelType = "rate_limited"):
        """
        根据模型类型返回对应的模型实例。

        如果模型尚未创建，会在首次调用时自动初始化并缓存。

        Args:
            model_type: 模型类型，可选值为 openai、tongyi、custom、rate_limited。

        Returns:
            Any: 对应的模型实例。

        Raises:
            ValueError: 当传入不支持的模型类型时抛出。
        """
        if model_type not in self._models:
            creators = {
                "openai": self._create_openai_model,
                "tongyi": self._create_tongyi_model,
                "custom": self._create_custom_model,
                "rate_limited": self._create_rate_limited_model,
            }

            if model_type not in creators:
                raise ValueError(f"不支持的模型类型: {model_type}")

            self._models[model_type] = creators[model_type]()

        return self._models[model_type]

    def _create_openai_model(self) -> ChatOpenAI:
        """创建 OpenAI 兼容模型实例。"""
        return ChatOpenAI(
            model="minimax-m2.5",
            temperature=0.5,
            api_key=API_KEY,
            base_url=BASE_URL,
        )

    def _create_tongyi_model(self) -> ChatOpenAI:
        """创建通义千问模型实例。"""
        return init_chat_model(
            model="qwen3.5-plus",
            model_provider="openai",
            api_key=TONGYI_API_KEY,
            base_url=TONGYI_BASE_URL,
        )

    def _create_custom_model(self) -> CustomLLM:
        """创建项目自定义封装模型实例。"""
        return CustomLLM(
            model="MiniMax-M2.5",
            api_key=MINIMAX_API_KEY,
            base_url=MINIMAX_BASE_URL,
        )

    def _create_rate_limited_model(self) -> ChatOpenAI:
        """
        创建带内存限流器的模型实例。

        该模型适合演示或低频调用场景，能够避免短时间内请求过于频繁。
        """
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=0.1,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

        return init_chat_model(
            model="MiniMax-M2.5",
            model_provider="openai",
            rate_limiter=rate_limiter,
            base_url=MINIMAX_BASE_URL,
            api_key=MINIMAX_API_KEY,
        )

    def invoke(self, prompt: str, model_type: ModelType = "rate_limited") -> str:
        """
        以非流式方式调用指定模型。

        Args:
            prompt: 用户输入的提示词。
            model_type: 要调用的模型类型。

        Returns:
            str: 模型返回的文本内容。
        """
        model = self.get_model(model_type)
        response = model.invoke(prompt)
        return getattr(response, "content", str(response))

    def stream(
        self, prompt: str, model_type: ModelType = "rate_limited"
    ) -> Iterator[str]:
        """
        以流式方式调用指定模型，并逐段返回生成内容。

        Args:
            prompt: 用户输入的提示词。
            model_type: 要调用的模型类型。

        Yields:
            Iterator[str]: 模型返回的文本片段。
        """
        model = self.get_model(model_type)
        for chunk in model.stream(prompt):
            content = getattr(chunk, "content", "")
            if content:
                yield content


if __name__ == "__main__":
    # 这里给出一个最小可运行示例，便于直接验证封装是否可用。
    service = ChatModelService()
    for text in service.stream("帮我讲个笑话吧", "tongyi"):
        print(text, end="")
