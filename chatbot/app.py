
# -*- coding: utf-8 -*-

from urllib.parse import urlsplit
import uuid

import httpx
import streamlit as st


# ==========================================
# 1. 기본 설정
# ==========================================

DATASETS = {
    "문서": "documents",
    "고객 요청": "inquiries",
}

MAX_CHARS = 10000

# 로컬 서버 환경을 고려한 Timeout 설정
API_TIMEOUT = httpx.Timeout(
    connect=5.0,    # 서버 연결 대기
    read=90.0,      # 응답 데이터 대기
    write=10.0,     # 요청 데이터 전송
    pool=5.0,       # 연결 풀 대기
)


# ==========================================
# 2. 사용자별 세션 초기화 (Isolation)
# ==========================================

def initialize_session():

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "last_exchange" not in st.session_state:
        st.session_state.last_exchange = None

    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = "문서"

    if "app_settings" not in st.session_state:
        st.session_state.app_settings = {
            "use_api": False,
            "api_url": "http://127.0.0.1:8000",
        }


# ==========================================
# 3. 질문·답변 박스 스타일
# ==========================================

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


# ==========================================
# 4. FastAPI 통신
# ==========================================

def get_reply(text, dataset):

    settings = st.session_state.app_settings

    # API 미연결 시 화면 테스트
    if not settings["use_api"]:
        return {
            "status": "demo",
            "classification": {
                "status": "skipped",
                "label": None,
                "confidence": None,
            },
            "generation": {
                "status": "success",
                "answer": (
                    "입력한 내용을 받았어요. "
                    "현재는 화면 테스트 중이며, "
                    "실제 분류와 답변 생성은 "
                    "모델 연결 후 제공됩니다."
                ),
                "next_steps": [],
            },
        }

    # 요청별 고유 ID 생성
    request_id = str(uuid.uuid4())

    # FastAPI 요청
    response = httpx.post(
        f"{settings['api_url'].rstrip('/')}/chat",
        json={
            "text": text,
            "dataset": dataset,
        },
        headers={
            "X-Request-ID": request_id,
        },
        timeout=API_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    # 응답 데이터 형식 확인
    if not isinstance(data, dict):
        raise ValueError(
            "API 응답 형식이 올바르지 않습니다."
        )

    if data.get("status") not in (
        "success",
        "partial_success",
    ):
        raise ValueError(
            "API 처리 상태를 확인할 수 없습니다."
        )

    if not isinstance(
        data.get("classification"), dict
    ):
        raise ValueError(
            "분류 결과가 없습니다."
        )

    if not isinstance(
        data.get("generation"), dict
    ):
        raise ValueError(
            "답변 생성 결과가 없습니다."
        )

    return data


# ==========================================
# 5. 분류 결과 및 LLM 답변 표시
# ==========================================

def show_reply(data):

    status = data.get("status")

    classification = data.get(
        "classification", {}
    )

    generation = data.get(
        "generation", {}
    )

    # 전체 요청 상태
    if status == "demo":
        st.info(
            "현재 화면 테스트 모드입니다."
        )

    elif status == "partial_success":
        st.warning(
            "일부 처리 과정에서 문제가 발생했습니다."
        )

    # --------------------------------------
    # 분류 결과
    # --------------------------------------

    classification_status = classification.get(
        "status"
    )

    if classification_status == "success":

        label = classification.get("label")
        confidence = classification.get(
            "confidence"
        )

        if isinstance(label, str):
            st.markdown(
                f"**분류 결과:** {label}"
            )

        if (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and 0 <= confidence <= 1
        ):
            st.markdown(
                f"**신뢰도:** {confidence:.1%}"
            )

    elif classification_status == "timeout":

        st.warning(
            "분류 모델 응답 시간이 초과되었습니다."
        )

    elif classification_status == "error":

        st.warning(
            "분류 모델 처리에 실패했습니다."
        )

    # --------------------------------------
    # LLM 답변
    # --------------------------------------

    generation_status = generation.get(
        "status"
    )

    if generation_status == "success":

        answer = generation.get("answer")

        if isinstance(answer, str) and answer.strip():

            st.markdown("**AI 답변**")

            st.write(answer)

        next_steps = generation.get(
            "next_steps", []
        )

        if isinstance(next_steps, list) and next_steps:

            st.markdown("**다음 단계**")

            for step in next_steps:
                if isinstance(step, str):
                    st.markdown(f"- {step}")

    elif generation_status == "timeout":

        st.warning(
            "AI 답변 생성 시간이 초과되었습니다. "
            "잠시 후 다시 시도해주세요."
        )

    elif generation_status == "error":

        st.warning(
            "AI 답변 생성에 실패했습니다. "
            "잠시 후 다시 시도해주세요."
        )

    elif generation_status == "skipped":

        if status != "demo":
            st.info(
                "분류 실패로 AI 답변 생성을 "
                "진행하지 않았습니다."
            )


# ==========================================
# 6. 데이터셋 변경
# ==========================================

def change_dataset():

    st.session_state.selected_dataset = (
        st.session_state._dataset
    )

    # 데이터셋 변경 시 이전 결과 초기화
    st.session_state.last_exchange = None


# ==========================================
# 7. 채팅 페이지
# ==========================================

def chat_page():

    apply_chat_style()

    title_col, button_col = st.columns([4, 1])

    with title_col:
        st.title("분류 챗봇")

    with button_col:
        if st.button("⚙️ 설정"):
            st.switch_page(settings_route)

    st.caption(
        "새 질문을 보내면 이전 질문과 답변이 교체됩니다."
    )

    if st.session_state.pop(
        "settings_saved", False
    ):
        st.success("설정을 저장했어요.")

    # 데이터셋 선택
    dataset_name = st.radio(
        "데이터셋",
        list(DATASETS),
        horizontal=True,
        index=list(DATASETS).index(
            st.session_state.selected_dataset
        ),
        key="_dataset",
        on_change=change_dataset,
    )

    if not st.session_state.app_settings["use_api"]:

        st.caption(
            "현재는 모델 없이 화면을 확인하는 "
            "테스트 모드입니다."
        )

    # 사용자 입력
    prompt = st.chat_input(
        "분류할 내용을 입력해주세요.",
        max_chars=MAX_CHARS,
    )

    if prompt is not None:

        prompt = prompt.strip()

        if not prompt or len(prompt) > MAX_CHARS:

            st.warning(
                "공백이 아닌 내용을 "
                "10,000자 이하로 입력해주세요."
            )

        else:

            # 현재 사용자 세션에 저장
            st.session_state.last_exchange = {
                "question": prompt,
                "answer": None,
                "dataset_name": dataset_name,
            }

            # ----------------------------------
            # API 실행 및 경과 시간 표시
            # ----------------------------------

            with st.spinner(
                "AI가 답변을 생성하고 있어요. "
                "잠시만 기다려주세요...",
                show_time=True,
            ):

                try:

                    result = get_reply(
                        prompt,
                        DATASETS[dataset_name],
                    )

                # 서버 연결 Timeout
                except httpx.ConnectTimeout:

                    result = {
                        "error": (
                            "서버 연결 시간이 초과되었습니다. "
                            "잠시 후 다시 시도해주세요."
                        )
                    }

                # 응답 대기 Timeout
                except httpx.ReadTimeout:

                    result = {
                        "error": (
                            "AI 응답 시간이 초과되었습니다. "
                            "잠시 후 다시 시도해주세요."
                        )
                    }

                # 그 외 Timeout
                except httpx.TimeoutException:

                    result = {
                        "error": (
                            "요청 처리 시간이 초과되었습니다. "
                            "잠시 후 다시 시도해주세요."
                        )
                    }

                # HTTP 오류
                except httpx.HTTPStatusError as exc:

                    result = {
                        "error": (
                            "API 요청에 실패했습니다. "
                            f"응답 코드: "
                            f"{exc.response.status_code}"
                        )
                    }

                # 네트워크 오류
                except httpx.RequestError:

                    result = {
                        "error": (
                            "FastAPI 서버에 연결하지 못했습니다. "
                            "API 주소와 서버 상태를 확인해주세요."
                        )
                    }

                # 응답 JSON 오류
                except (ValueError, TypeError) as exc:

                    result = {
                        "error": str(exc)
                    }

            # 사용자 세션에 결과 저장
            st.session_state.last_exchange["answer"] = result

    # --------------------------------------
    # 질문 및 답변 출력
    # --------------------------------------

    exchange = st.session_state.last_exchange

    if exchange is not None:

        # 질문: 파란 음영 박스
        with st.container(
            key="question_box",
            border=False,
        ):

            st.caption("👤 질문")
            st.text(exchange["question"])

        # 답변: 회색 음영 박스
        with st.container(
            key="answer_box",
            border=False,
        ):

            st.caption(
                f"🤖 답변 · 데이터셋: "
                f"{exchange['dataset_name']}"
            )

            result = exchange["answer"]

            if isinstance(result, dict):

                if "error" in result:
                    st.error(result["error"])

                else:
                    show_reply(result)


# ==========================================
# 8. 설정 페이지
# ==========================================

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
            "모델이 준비되기 전에는 API 연결을 끄고 "
            "화면을 테스트하세요."
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
                "API 주소를 http:// 또는 https://로 "
                "시작하는 주소로 입력해주세요."
            )

        else:

            updated = {
                "use_api": use_api,
                "api_url": api_url,
            }

            # 설정 변경 시 이전 결과 초기화
            if updated != saved:
                st.session_state.last_exchange = None

            st.session_state.app_settings = updated

            st.session_state.settings_saved = True

            st.switch_page(chat_route)


# ==========================================
# 9. Streamlit 실행 및 초기화
# ==========================================

st.set_page_config(
    page_title="분류 챗봇",
    page_icon="💬",
    layout="centered",
)

initialize_session()


# ==========================================
# 10. 페이지 등록 및 라우팅
# ==========================================

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
