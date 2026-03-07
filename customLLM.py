import os
from typing import Any, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from openai import OpenAI

from env_utils import TONGYI_API_KEY, TONGYI_BASE_URL


class CustomLLM(BaseChatModel):
    """
    自定义 LLM 聊天模型类，基于 OpenAI 兼容接口实现。

    该类继承自 LangChain 的 BaseChatModel，封装了对外部大语言模型的调用逻辑。
    支持通义千问等兼容 OpenAI 接口的模型，提供消息转换、流式响应、推理内容提取等功能。

    主要功能：
    - 将 LangChain 消息格式转换为 OpenAI API 格式
    - 支持系统消息、用户消息、助手消息的处理
    - 可选启用模型的推理能力（enable_thing）
    - 捕获并返回模型的推理过程（reasoning_content）
    - 统一的错误处理和异常抛出

    属性说明：
        model: 模型名称，默认为 "qwen-max"
        api_key: API 密钥，默认从环境变量读取
        base_url: API 基础 URL，默认从环境变量读取
        enable_thing: 是否启用推理功能，默认为 True
        use_stream: 是否偏好使用流式接口的配置开关，默认 False
        temperature: 温度参数，控制输出随机性，范围 0-1，默认 0.7
        max_tokens: 最大生成 token 数，默认 2048

    使用示例：
        llm = CustomLLM(
            model="qwen-max",
            temperature=0.7,
            enable_thing=True
        )
        result = llm.invoke("你好，请介绍一下你自己")
        print(result.content)
    """

    # 模型名称，指定要使用的大语言模型
    model: str = "qwen-max"

    # API 密钥，用于身份验证
    api_key: str = TONGYI_API_KEY

    # API 基础 URL，指向模型服务的端点
    base_url: str = TONGYI_BASE_URL

    # 是否启用推理功能（thinking），某些模型支持返回推理过程
    enable_thing: bool = True

    # 是否偏好使用流式接口。
    # 注意：不能命名为 stream，避免遮蔽父类的 stream() 方法。
    use_stream: bool = False

    # 温度参数，控制输出的随机性和创造性，值越高越随机
    temperature: float = 0.7

    # 最大生成的 token 数量，限制输出长度
    max_tokens: int = 2048

    @property
    def _llm_type(self) -> str:
        """
        返回 LLM 类型标识符。

        LangChain 框架要求实现此属性，用于标识模型类型。

        Returns:
            str: 包含模型名称和基础 URL 的描述字符串
        """
        return f"模型名称: {self.model}, 基础URL: {self.base_url}"

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """将 LangChain 消息转换为 OpenAI 兼容消息格式。"""
        openai_messages: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                openai_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                openai_messages.append(
                    {"role": "assistant", "content": message.content}
                )
            elif isinstance(message, SystemMessage):
                openai_messages.append({"role": "system", "content": message.content})
        return openai_messages

    def _create_client(self) -> OpenAI:
        """创建 OpenAI 兼容客户端。"""
        return OpenAI(
            api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self.base_url,
        )

    def _extract_reasoning_content(self, payload: Any) -> str:
        """尽力从响应对象中提取 reasoning_content，提取不到时返回空字符串。"""
        if payload is None:
            return ""

        model_extra = getattr(payload, "model_extra", None)
        if isinstance(model_extra, dict):
            reasoning = model_extra.get("reasoning_content")
            if isinstance(reasoning, str):
                return reasoning

        if isinstance(payload, dict):
            reasoning = payload.get("reasoning_content")
            if isinstance(reasoning, str):
                return reasoning

        return ""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        生成聊天响应的核心方法（非流式）。

        该方法是 LangChain BaseChatModel 要求实现的抽象方法，负责：
        1. 将 LangChain 消息格式转换为 OpenAI API 格式
        2. 调用外部模型 API 获取响应（非流式）
        3. 解析响应并提取推理内容（如果启用）
        4. 将响应封装为 LangChain 标准格式返回

        注意：此方法不处理流式响应。流式响应由 _stream 方法处理。
        虽然设置了 stream 参数，但在此方法中始终作为非流式处理。

        Args:
            messages: LangChain 消息列表，包含对话历史
            stop: 停止词列表，遇到这些词时停止生成（当前未使用）
            run_manager: LangChain 回调管理器，用于追踪执行过程（当前未使用）
            **kwargs: 其他关键字参数

        Returns:
            ChatResult: 包含生成结果的 LangChain 标准响应对象

        Raises:
            ValueError: 当模型返回内容为空或 API 调用失败时抛出
        """
        # 第一步：将 LangChain 消息格式转换为 OpenAI API 格式
        openai_messages = self._convert_messages(messages)

        # 第二步：创建 OpenAI 客户端
        client = self._create_client()

        # 第三步：构建 API 调用参数（_generate 方法强制非流式）
        call_params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        # 如果启用推理功能，添加额外参数
        if self.enable_thing:
            call_params["extra_body"] = {"enable_thing": True}

        try:
            # 第四步：调用模型 API
            completion = client.chat.completions.create(**call_params)
            rc = ""
            message = completion.choices[0].message if completion.choices else None
            content = getattr(message, "content", None)
            if message and content:
                rc = self._extract_reasoning_content(message)
                aimessage = AIMessage(
                    content=content,
                    additional_kwargs={
                        "model": self.model,
                        "usage": getattr(completion, "usage", {}),
                        "reasoning_content": rc,
                    },
                )
                return ChatResult(generations=[ChatGeneration(message=aimessage)])
            raise ValueError("模型返回内容为空")
        except Exception as err:
            raise ValueError(f"调用模型接口出错: {str(err)}") from err

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        生成流式聊天响应的方法。

        该方法实现了真正的流式处理，能够实时返回模型生成的内容片段。
        用户可以在模型生成过程中就看到部分结果，提升交互体验。

        流式处理流程：
        1. 将 LangChain 消息格式转换为 OpenAI API 格式
        2. 调用 API 并设置 stream=True
        3. 循环接收并处理每个数据块（chunk）
        4. 逐步累积推理内容
        5. 将每个 chunk 封装为 ChatGenerationChunk 并 yield 返回

        Args:
            messages: LangChain 消消息列表，包含对话历史
            stop: 停止词列表，遇到这些词时停止生成（当前未使用）
            run_manager: LangChain 回调管理器，用于追踪执行过程
            **kwargs: 其他关键字参数

        Yields:
            ChatGenerationChunk: 每个生成的内容片段

        Raises:
            ValueError: 当 API 调用失败时抛出
        """
        # 第一步：将 LangChain 消息格式转换为 OpenAI API 格式
        openai_messages = self._convert_messages(messages)

        # 第二步：创建 OpenAI 客户端
        client = self._create_client()

        # 第三步：构建 API 调用参数（流式模式）
        call_params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        # 如果启用推理功能，添加额外参数
        if self.enable_thing:
            call_params["extra_body"] = {"enable_thing": True}

        try:
            # 第四步：调用模型 API（流式）
            stream = client.chat.completions.create(**call_params)

            # 用于累积推理内容
            reasoning_content = ""
            # 第五步：循环处理流式响应
            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue

                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                content = getattr(delta, "content", None) or ""
                rc_chunk = self._extract_reasoning_content(delta)
                if rc_chunk:
                    reasoning_content += rc_chunk

                if not content and not rc_chunk:
                    continue

                chunk_message = AIMessageChunk(
                    content=content,
                    additional_kwargs={
                        "model": self.model,
                        "reasoning_content": reasoning_content,
                    },
                )
                generation_chunk = ChatGenerationChunk(message=chunk_message)
                yield generation_chunk

                if run_manager and content:
                    run_manager.on_llm_new_token(content, chunk=generation_chunk)
        except Exception as err:
            raise ValueError(f"流式调用模型接口出错: {str(err)}") from err
