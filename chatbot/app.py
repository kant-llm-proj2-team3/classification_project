# -*- coding: utf-8 -*-

from urllib.parse import urlsplit

import httpx
import streamlit as st


DATASETS = {
    "문서": "documents",
    "고객 요청": "inquiries",
}

MAX_CHARS = 10000
MODEL_ID = "demo"


# 질문·답변 박스 스타일
def apply_chat_style():
    st.html(
        """
        <style>
        .st-key-question_box,
        .st-key-answer_box {
            padding: 20px 24px;
            border-radius: 18px;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.07);
            color: #1f2937;
        }

        .st-key-question_box {
            background-color: #eaf3ff;
            border: 1px solid #cbdff8;
        }

        .st-key-answer_box {
            background-color: #f3f4f6;
            border: 1px solid #e0e3e8;
        }

        .st-key-question_box [data-testid="stText"],
        .st-key-answer_box [data-testid="stText"] {
            color: #1f2937;
            line-height: 1.7;
            overflow-wrap: anywhere;
        }

        .st-key-question_box [data-testid="stCaptionContainer"],
        .st-key-answer_box [data-testid="stCaptionContainer"] {
            color: #4b5563;
        }
        </style>
        """
    )


# API 요청 또는 화면 테스트용 응답
def get_reply(text, dataset):
    settings = st.session_state.app_settings

    if not settings["use_api"]:
        return (
            "입력한 내용을 받았어요. 현재는 화면 테스트 중이며, "
            "실제 분류는 모델 연결 후 제공됩니다."
        )

    response = httpx.post(
        f"{settings['api_url'].rstrip('/')}/classify",
        json={
            "text": text,
            "dataset": dataset,
            "model_id": MODEL_ID,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, dict) or not isinstance(data.get("label"), str):
        raise ValueError("API 응답에 분류 결과(label)가 없습니다.")

    parts = [f"분류 결과: {data['label']}"]

    if data.get("is_demo"):
        parts.append("※ 데모 응답입니다. 실제 분류 결과가 아닙니다.")

    confidence = data.get("confidence")

    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        parts.append(f"신뢰도: {confidence:.1%}")

    return "\n\n".join(parts)


# 데이터셋을 바꾸면 이전 질문·답변 초기화
def change_dataset():
    st.session_state.selected_dataset = st.session_state._dataset
    st.session_state.last_exchange = None


# 채팅 페이지
def chat_page():
    apply_chat_style()

    title_col, button_col = st.columns([4, 1])

    with title_col:
        st.title("분류 챗봇")

    with button_col:
        if st.button("⚙️ 설정"):
            st.switch_page(settings_route)

    st.caption("새 질문을 보내면 이전 질문과 답변이 교체됩니다.")

    if st.session_state.pop("settings_saved", False):
        st.success("설정을 저장했어요.")

    dataset_name = st.radio(
        "데이터셋",
        list(DATASETS),
        horizontal=True,
        index=list(DATASETS).index(st.session_state.selected_dataset),
        key="_dataset",
        on_change=change_dataset,
    )

    if not st.session_state.app_settings["use_api"]:
        st.caption("현재는 모델 없이 화면을 확인하는 테스트 모드입니다.")

    prompt = st.chat_input(
        "분류할 내용을 입력해주세요.",
        max_chars=MAX_CHARS,
    )

    if prompt is not None:
        prompt = prompt.strip()

        if not prompt or len(prompt) > MAX_CHARS:
            st.warning("공백이 아닌 내용을 10,000자 이하로 입력해주세요.")

        else:
            # 가장 최근 질문과 답변 한 쌍만 저장
            st.session_state.last_exchange = {
                "question": prompt,
                "answer": "",
                "dataset_name": dataset_name,
            }

            with st.spinner("내용을 확인하고 있어요..."):
                try:
                    reply = get_reply(
                        prompt,
                        DATASETS[dataset_name],
                    )

                except httpx.TimeoutException:
                    reply = "응답 시간이 초과됐어요. 잠시 후 다시 시도해주세요."

                except httpx.HTTPStatusError as exc:
                    reply = (
                        "분류 요청에 실패했어요. "
                        f"응답 코드: {exc.response.status_code}"
                    )

                except httpx.RequestError:
                    reply = (
                        "모델 API에 연결하지 못했어요. "
                        "설정에서 API 주소를 확인해주세요."
                    )

                except ValueError as exc:
                    reply = str(exc)

            st.session_state.last_exchange["answer"] = reply

    exchange = st.session_state.last_exchange

    if exchange is not None:
        # 질문: 파란 박스
        with st.container(key="question_box", border=False):
            st.caption("👤 질문")
            st.text(exchange["question"])

        # 답변: 회색 박스
        with st.container(key="answer_box", border=False):
            st.caption(f"🤖 답변 · 데이터셋: {exchange['dataset_name']}")
            st.text(exchange["answer"])


# 설정 페이지
def settings_page():
    st.title("⚙️ 설정")

    if st.button("← 채팅으로 돌아가기"):
        st.switch_page(chat_route)

    saved = st.session_state.app_settings

    with st.form("settings_form"):
        use_api = st.checkbox(
            "모델 API 연결",
            value=saved["use_api"],
        )

        api_url = st.text_input(
            "API 주소",
            value=saved["api_url"],
        )

        st.caption(
            "모델이 준비되기 전에는 API 연결을 끄고 화면을 테스트하세요."
        )

        submitted = st.form_submit_button(
            "저장하고 채팅으로 돌아가기"
        )

    if submitted:
        api_url = api_url.strip().rstrip("/")

        try:
            parsed = urlsplit(api_url)
            valid_url = (
                parsed.scheme in ("http", "https")
                and bool(parsed.hostname)
            )

        except ValueError:
            valid_url = False

        if use_api and not valid_url:
            st.error(
                "API 주소를 http:// 또는 https://로 시작하는 주소로 입력해주세요."
            )

        else:
            updated = {
                "use_api": use_api,
                "api_url": api_url,
            }

            # 설정이 바뀌면 이전 결과 초기화
            if updated != saved:
                st.session_state.last_exchange = None

            st.session_state.app_settings = updated
            st.session_state.settings_saved = True

            st.switch_page(chat_route)


# 공통 설정과 세션 초기화
st.set_page_config(
    page_title="분류 챗봇",
    page_icon="💬",
    layout="centered",
)

st.session_state.pop("messages", None)

if "last_exchange" not in st.session_state:
    st.session_state.last_exchange = None

if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = "문서"

if "app_settings" not in st.session_state:
    st.session_state.app_settings = {
        "use_api": False,
        "api_url": "http://127.0.0.1:8000",
    }


# 페이지 등록과 라우팅
chat_route = st.Page(
    chat_page,
    title="채팅",
    default=True,
)

settings_route = st.Page(
    settings_page,
    title="설정",
    url_path="settings",
)

current_page = st.navigation(
    [chat_route, settings_route],
    position="hidden",
)

current_page.run()