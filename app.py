import streamlit as st
from openai import OpenAI
from openai import APIError, APIConnectionError, AuthenticationError
from PyPDF2 import PdfReader
import pandas as pd


# ===================== 页面设置 =====================
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

# ---- 自定义样式 ----
st.markdown("""
<style>
    /* 全局字体与背景 */
    html, body, [class*="css"] {
        font-family: "Inter", "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    /* 主标题 */
    h1 { color: #1a73e8; font-weight: 700; }
    /* 侧边栏 */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #e8f0fe 0%, #d2e3fc 100%);
    }
    /* 按钮 */
    .stButton > button {
        border-radius: 8px; border: none;
        background: #1a73e8; color: white; font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover { background: #1557b0; transform: scale(1.02); }
    /* 输入框 */
    input, textarea { border-radius: 8px !important; }
    /* 聊天气泡 */
    [data-testid="stChatMessage"] {
        border-radius: 12px; padding: 12px 16px; margin: 6px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 张梓源的AI Chatbot")

# ===================== 初始化 session state =====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_content" not in st.session_state:
    st.session_state.uploaded_file_content = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# ===================== 侧边栏 =====================
with st.sidebar:
    st.header("⚙️ API 配置")

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-xxxxxxxxxxxxxxxx",
        help="LLM 的 API Key",
    )

    base_url = st.text_input(
        "Base URL",
        value="https://api.openai.com/v1",
        help="兼容 OpenAI 的 API 端点",
    )

    model = st.text_input(
        "Model",
        value="gpt-4o",
        help="模型名称",
    )

    st.divider()

    # ---------- 模型参数 ----------
    st.header("🔧 模型参数")

    context_rounds = st.slider(
        "上下文轮数",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="保留最近 N 轮对话发送给模型（1 轮 = 一问一答）",
    )

    max_tokens = st.slider(
        "最大输出长度 (max_tokens)",
        min_value=256,
        max_value=16384,
        value=4096,
        step=256,
        help="模型单次回复的最大 token 数，控制回复长度",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="越低越确定，越高越随机",
    )

    st.divider()

    # ---------- 文件上传 ----------
    st.header("📎 文件上传")

    uploaded_file = st.file_uploader(
        "支持 TXT / PDF / CSV",
        type=["txt", "pdf", "csv"],
    )

    if uploaded_file is not None:
        try:
            file_type = uploaded_file.name.split(".")[-1].lower()

            if file_type == "txt":
                content = uploaded_file.read().decode("utf-8", errors="ignore")
            elif file_type == "pdf":
                pdf_reader = PdfReader(uploaded_file)
                content = ""
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
            elif file_type == "csv":
                df = pd.read_csv(uploaded_file)
                content = df.to_string(max_rows=200)
            else:
                content = None

            if content:
                max_chars = 15000
                if len(content) > max_chars:
                    content = content[:max_chars] + f"\n\n…（文件过长，已截取前 {max_chars} 字符）"
                st.session_state.uploaded_file_content = content
                st.session_state.uploaded_file_name = uploaded_file.name
                st.success(f"✅ 已读取: {uploaded_file.name}（{len(content)} 字符）")

        except Exception as e:
            st.error(f"❌ 读取文件失败: {e}")

    if st.session_state.uploaded_file_content:
        if st.button("🗑️ 清除已上传文件", use_container_width=True):
            st.session_state.uploaded_file_content = None
            st.session_state.uploaded_file_name = None
            st.rerun()

    st.divider()

    # ---------- 清空对话 ----------
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ===================== 显示历史消息 =====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg.get("display", msg["content"]))

# ===================== 文件提示条 =====================
if st.session_state.uploaded_file_name:
    st.info(f"📎 当前附加文件: **{st.session_state.uploaded_file_name}**（文件内容会自动加入每次对话上下文）")

# ===================== 用户输入 =====================
if prompt := st.chat_input("输入你的问题…"):
    # ---- 构建完整的用户消息 ----
    if st.session_state.uploaded_file_content:
        user_content = (
            f"【用户上传文件: {st.session_state.uploaded_file_name}】\n"
            f"{st.session_state.uploaded_file_content}\n\n"
            f"【用户问题】\n{prompt}"
        )
    else:
        user_content = prompt

    # ---- 显示用户消息 ----
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---- 保存到历史 ----
    st.session_state.messages.append({
        "role": "user",
        "content": user_content,
        "display": prompt,
    })

    # ---- 检查必填项 ----
    if not api_key:
        with st.chat_message("assistant"):
            st.warning("⚠️ 请先在侧边栏填写 API Key")
        st.stop()

    # ---- 调用 API ----
    with st.chat_message("assistant"):
        try:
            max_messages = context_rounds * 2
            recent_messages = st.session_state.messages[-max_messages:]

            client = OpenAI(api_key=api_key, base_url=base_url)
            stream = client.chat.completions.create(
                model=model,
                messages=recent_messages,
                stream=True,
                max_tokens=max_tokens,
                temperature=temperature,
            )

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
            st.error("❌ 无法连接到 API 服务器，请检查 Base URL 是否拼写正确")
            st.stop()
        except APIError as e:
            st.error(f"❌ API 错误：{e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ 未知错误：{e}")
            st.stop()
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
            })
