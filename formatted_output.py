from langchain_core.output_parsers import SimpleJsonOutputParser
from pydantic import BaseModel, Field

from chat_model_service import ChatModelService


class Movie(BaseModel):
    title: str = Field(..., description="电影标题")
    director: str = Field(..., description="电影导演")
    release_year: int = Field(..., description="电影上映年份")
    rating: float = Field(..., description="电影评分")


class FormattedOutputService(ChatModelService):
    """
    这是一个示例服务类，展示如何使用 ChatModelService 来获取结构化输出。
    """

    def __init__(self):
        super().__init__()
        self.llm = self.get_model("tongyi")

    def structured_output(self, query: str):
        # 使用 with_structured_output 方法指定输出结构为 Movie
        output = self.llm.with_structured_output(Movie).invoke(query)
        return output

    def simple_json_output(self, query: str):
        # 使用 with_structured_output 方法指定输出结构为 dict
        output = self.llm.with_structured_output(dict).invoke(query)
        return output


if __name__ == "__main__":
    service = ChatModelService()
    # 不使用 with_structured_output，直接用普通方式调用
    llm = service.get_model("tongyi")

    # 在 prompt 中明确要求 JSON 格式输出
    query = "提供电影盗墓空间的详细信息"

    output = query | llm | SimpleJsonOutputParser()

    print(output)
