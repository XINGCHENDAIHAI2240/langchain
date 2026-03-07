from langchain_core.prompts import PromptTemplate

from chat_model_service import ChatModelService

prompt_template = PromptTemplate.from_template(
    "我的领居姓{last_name}，刚生了{gender},请帮我起个名字，名字为三个字，简单回答？"
)

# 调用.format 方法注入信息即可
# prompt_template.format(last_name="喻", gender="男")

llm = ChatModelService().get_model()
chain = prompt_template | llm

if __name__ == "__main__":
    res = chain.invoke({"last_name": "喻", "gender": "男"})
    print(res.content)
