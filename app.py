import streamlit as st
from openai import OpenAI
from openai import APIError, APIConnectionError, AuthenticationError


# ---------- 页面设置 ----------
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot")

# ---------- 侧边栏：API 配置 ----------
with st.sidebar:
    st.header("⚙️ API 配置")

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-xxxxxxxxxxxxxxxx",
        help="输入你的 API Key，不会存储在服务器上",
    )

    base_url = st.text_input(
        "Base URL",
        value="https://api.openai.com/v1",
        help="API 端点地址。支持任何兼容 OpenAI 的 API（如 DeepSeek、Ollama 等）",
    )

    model = st.text_input(
        "Model",
        value="gpt-4o",
        help="模型名称，如 gpt-4o、gpt-3.5-turbo、deepseek-chat 等",
    )

    st.divider()

    # 清空对话按钮
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- 初始化聊天历史 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- 显示历史消息 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 用户输入 ----------
if prompt := st.chat_input("输入你的问题…"):
    # 加入用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 检查 API Key
    if not api_key:
        with st.chat_message("assistant"):
            st.warning("⚠️ 请先在侧边栏填写 API Key")
        st.stop()

    # 调用 API 获取回复
    with st.chat_message("assistant"):
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            stream = client.chat.completions.create(
                model=model,
                messages=st.session_state.messages,
                stream=True,
            )

            # 流式输出：逐个 token 显示
            response = ""
            placeholder = st.empty()
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    response += delta.content
                    placeholder.markdown(response + "▌")
            placeholder.markdown(response)

        except AuthenticationError:
            st.error("❌ API Key 无效，请检查后重试")
            st.stop()
        except APIConnectionError:
            st.error("❌ 无法连接到 API 服务器，请检查 Base URL")
            st.stop()
        except APIError as e:
            st.error(f"❌ API 错误：{e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ 未知错误：{e}")
            st.stop()
        else:
            # 只有成功时才保存 AI 回复
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
