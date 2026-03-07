from openai import OpenAI

from env_utils import API_KEY, BASE_URL

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# response = client.chat.completions.create(
#     model = "minmax-m2.5",
#     messages=[
#         {
#             "role": "user",
#             "content": "你是一个Python专家，帮我解答问题。"
#         },
#         {
#             "role": "assistant",
#             "content": "我是一个Python专家，请问有什么问题需要我解答吗？"
#         },
#         {
#             "role": "user",
#             "content": "请帮我写一个Python函数，输入一个字符串，返回字符串的反转。"
#         }
#     ]
# )
#
# print(response.choices[0].message.content)

# 流式输出
# completion = client.chat.completions.create(
#     model="minimax-m2.5",
#     messages=[
#         {
#             "role": "user",
#             "content": "你是一个Python专家，帮我解答问题。"
#         },
#         {
#             "role": "assistant",
#             "content": "我是一个Python专家，请问有什么问题需要我解答吗？"
#         },
#         {
#             "role": "user",
#             "content": "请帮我写一个Python函数，输入一个字符串，返回字符串的反转。"}
#     ],
#     stream=True
# )

completion = client.chat.completions.create(
    model="minimax-m2.5",
    messages=[{"role": "user", "content": "帮我讲个笑话吧"}],
    stream=True,
)

for message in completion:
    print(message.choices[0].delta.content, end="", flush=True)
