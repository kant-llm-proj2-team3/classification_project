# -*- coding: utf-8 -*-

from urllib.parse import urlsplit
import uuid

import httpx
import streamlit as st

import tracing_page


# ==========================================
# 1. 기본 설정
# ==========================================

DATASETS = {
    "문서": "documents",
    "고객 요청": "inquiries",
}

# 답변을 만들 생성 모델. 비교는 같은 입력을 두 모델에 동시에 보내 답변을 나란히 보여준다.
GENERATION_MODES = {
    "OpenAI": "openai",
    "로컬 모델 (Ollama)": "ollama",
    "비교 (둘 다)": "compare",
}

PROVIDER_NAMES = {
    "openai": "OpenAI",
    "ollama": "로컬 모델 (Ollama)",
}

MAX_CHARS = 10000

API_TIMEOUT = httpx.Timeout(
    connect=5.0,   # 서버 연결 대기
    read=90.0,     # 응답 데이터 대기
    write=10.0,    # 요청 데이터 전송
    pool=5.0,      # 연결 풀 대기
)


# ==========================================
# 2. 사용자별 세션 초기화
# ==========================================

def initialize_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "last_exchange" not in st.session_state:
        st.session_state.last_exchange = None

    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = "문서"

    if "generation_mode" not in st.session_state:
        st.session_state.generation_mode = "OpenAI"

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

def validate_api_url(api_url):
    """채팅과 연결 확인에 사용할 서버 주소를 검증한다."""
    try:
        parsed = urlsplit(api_url)

        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.hostname)
            and parsed.port != 0
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
        )

    except ValueError:
        return False


def check_api_connection(api_url):
    """모델 실행 여부와 별개로 FastAPI 서버 연결을 확인한다."""
    response = httpx.get(
        f"{api_url.rstrip('/')}/health",
        timeout=httpx.Timeout(5.0),
    )

    response.raise_for_status()
    data = response.json()

    if not isinstance(data, dict) or data.get("status") != "ok":
        raise ValueError(
            "서버 상태 응답 형식이 올바르지 않습니다."
        )

    return data


ALLOWED_STATUSES = {
    "success",
    "skipped",
    "error",
    "timeout",
}


def is_text_list(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def validate_comparison(data):
    """/compare 응답(분류 결과 하나 + 생성 모델별 결과 목록)을 확인한다."""
    if not isinstance(data, dict):
        raise ValueError("API 응답 형식이 올바르지 않습니다.")

    if data.get("status") not in ("success", "partial_success"):
        raise ValueError("API 처리 상태를 확인할 수 없습니다.")

    classification = data.get("classification")

    if (
        not isinstance(classification, dict)
        or classification.get("status") not in ALLOWED_STATUSES
    ):
        raise ValueError("분류 결과가 올바르지 않습니다.")

    results = data.get("results")

    if not isinstance(results, list) or not results:
        raise ValueError("답변 생성 결과가 없습니다.")

    for result in results:
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("provider"), str)
            or result.get("status") not in ALLOWED_STATUSES
        ):
            raise ValueError("API 답변 생성 결과의 형식이 올바르지 않습니다.")

        if result["status"] == "success":
            steps = result.get("next_steps", [])

            summary_items = result.get("summary_items", [])

            if (
                not isinstance(result.get("answer"), str)
                or not result["answer"].strip()
                or not is_text_list(steps)
                or not is_text_list(summary_items)
            ):
                raise ValueError("API 답변 내용이 올바르지 않습니다.")

    return data


def get_reply(text, dataset, mode="openai"):
    settings = st.session_state.app_settings

    # API 미연결 시에는 화면 테스트용 응답을 반환
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

    request_id = str(uuid.uuid4())

    # 선택한 데이터셋과 입력 내용을 FastAPI에 전송
    # 기본(OpenAI)은 예전과 같은 요청 형식이라 이전 버전 서버에서도 동작한다.
    payload = {
        "text": text,
        "dataset": dataset,
    }

    if mode == "compare":
        path = "/compare"

    else:
        path = "/chat"

        if mode != "openai":
            payload["provider"] = mode

    response = httpx.post(
        f"{settings['api_url'].rstrip('/')}{path}",
        json=payload,
        headers={
            "X-Request-ID": request_id,
        },
        timeout=API_TIMEOUT,
    )

    response.raise_for_status()
    data = response.json()

    if mode == "compare":
        return validate_comparison(data)

    # 응답 형식 검증
    if not isinstance(data, dict):
        raise ValueError(
            "API 응답 형식이 올바르지 않습니다."
        )

    if data.get("status") not in ("success", "partial_success"):
        raise ValueError(
            "API 처리 상태를 확인할 수 없습니다."
        )

    if not isinstance(data.get("classification"), dict):
        raise ValueError(
            "분류 결과가 없습니다."
        )

    if not isinstance(data.get("generation"), dict):
        raise ValueError(
            "답변 생성 결과가 없습니다."
        )

    allowed_statuses = {
        "success",
        "skipped",
        "error",
        "timeout",
    }

    for name in ("classification", "generation"):
        if data[name].get("status") not in allowed_statuses:
            raise ValueError(
                "API 처리 단계의 상태가 올바르지 않습니다."
            )

    generation = data["generation"]

    if generation["status"] == "success":
        answer = generation.get("answer")

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                "API 답변 내용이 없습니다."
            )

        steps = generation.get("next_steps", [])

        if (
            not isinstance(steps, list)
            or any(not isinstance(step, str) for step in steps)
        ):
            raise ValueError(
                "API 다음 단계 형식이 올바르지 않습니다."
            )

        # 문서(documents)는 항목형 요약(summary_items)을 함께 돌려준다.
        summary_items = generation.get("summary_items")

        if summary_items is not None and not is_text_list(summary_items):
            raise ValueError(
                "API 요약 항목 형식이 올바르지 않습니다."
            )

    return data


# ==========================================
# 5. 분류 결과 및 답변 표시
# ==========================================

def show_summary_items(items):
    """문서 요약의 항목('항목명: 내용')을 항목 이름을 굵게 해서 목록으로 보여준다."""
    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue

        name, separator, content = item.partition(":")

        if separator and 0 < len(name.strip()) <= 20:
            st.markdown(f"- **{name.strip()}**: {content.strip()}")

        else:
            st.markdown(f"- {item.strip()}")


def show_reply(data):
    status = data.get("status")

    is_stub = data.get("mode") == "stub"
    is_preview = is_stub or status == "demo"

    classification = data.get("classification", {})
    generation = data.get("generation", {})

    # 전체 요청 상태
    if status == "demo":
        st.info(
            "현재 화면 테스트 모드입니다."
        )

    elif is_stub:
        st.info(
            "FastAPI 연결이 확인되었습니다. "
            "현재는 연결 테스트용 응답이며, "
            "실제 분류 모델과 LLM은 아직 연결되지 않았습니다."
        )

    elif status == "partial_success":
        st.warning(
            "일부 처리 과정에서 문제가 발생했습니다."
        )

    # 분류 결과 표시
    classification_status = classification.get("status")

    if classification_status == "success":
        label = classification.get("label")
        confidence = classification.get("confidence")

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

    elif classification_status == "skipped" and not is_preview:
        st.info(
            "분류를 수행하지 않았습니다."
        )

    # 답변 표시
    generation_status = generation.get("status")

    if generation_status == "success":
        answer = generation.get("answer")

        summary_items = generation.get("summary_items")
        has_summary = isinstance(summary_items, list) and bool(summary_items)

        if isinstance(answer, str) and answer.strip():
            if is_stub:
                answer_title = "**연결 테스트 안내**"
            elif is_preview:
                answer_title = "**화면 테스트 안내**"
            elif has_summary:
                answer_title = "**문서 요약**"
            else:
                answer_title = "**AI 답변**"

            st.markdown(answer_title)
            st.write(answer)

        if has_summary:
            st.markdown("**항목별 요약**")
            show_summary_items(summary_items)

        next_steps = generation.get("next_steps", [])

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
                "분류를 수행하지 못해 AI 답변 생성을 "
                "진행하지 않았습니다."
            )


def show_comparison(data):
    """같은 입력과 프롬프트로 만든 생성 모델별 답변을 나란히 보여준다."""
    if data.get("status") == "partial_success":
        st.warning(
            "일부 처리 과정에서 문제가 발생했습니다."
        )

    classification = data["classification"]

    if classification["status"] == "success":
        label = classification.get("label")
        confidence = classification.get("confidence")

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

    elif classification["status"] == "timeout":
        st.warning("분류 모델 응답 시간이 초과되었습니다.")

    else:
        st.warning("분류 모델 처리에 실패했습니다.")

    results = data["results"]

    for column, result in zip(st.columns(len(results)), results):
        provider = result["provider"]
        metrics = result.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}

        with column:
            st.markdown(
                f"**{PROVIDER_NAMES.get(provider, provider)}**"
            )

            facts = []

            if isinstance(metrics.get("model"), str):
                facts.append(metrics["model"])

            if isinstance(metrics.get("latency_ms"), (int, float)):
                facts.append(f"{metrics['latency_ms'] / 1000:.1f}초")

            if isinstance(metrics.get("output_tokens"), int):
                facts.append(f"출력 {metrics['output_tokens']:,}토큰")

            if facts:
                st.caption(" · ".join(facts))

            if result["status"] == "success":
                st.write(result["answer"])

                summary_items = result.get("summary_items") or []

                if summary_items:
                    st.markdown("**항목별 요약**")
                    show_summary_items(summary_items)

                steps = result.get("next_steps") or []

                if steps:
                    st.markdown("**다음 단계**")

                    for step in steps:
                        st.markdown(f"- {step}")

            elif result["status"] == "timeout":
                st.warning(
                    "AI 답변 생성 시간이 초과되었습니다."
                )

            elif result["status"] == "skipped":
                st.info(
                    "분류를 수행하지 못해 AI 답변 생성을 "
                    "진행하지 않았습니다."
                )

            else:
                st.warning(
                    "AI 답변 생성에 실패했습니다."
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


def change_generation_mode():
    st.session_state.generation_mode = (
        st.session_state._generation_mode
    )

    # 생성 모델 변경 시 이전 결과 초기화
    st.session_state.last_exchange = None


# ==========================================
# 7. 채팅 페이지
# ==========================================

def chat_page():
    apply_chat_style()

    st.title("분류 챗봇")

    st.caption(
        "새 질문을 보내면 이전 질문과 답변이 교체됩니다."
    )

    if st.session_state.pop("settings_saved", False):
        st.success("설정을 저장했어요.")

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

    mode_name = st.radio(
        "답변 생성 모델",
        list(GENERATION_MODES),
        horizontal=True,
        index=list(GENERATION_MODES).index(
            st.session_state.generation_mode
        ),
        key="_generation_mode",
        on_change=change_generation_mode,
    )

    if mode_name == "비교 (둘 다)":
        st.caption(
            "같은 입력을 OpenAI와 로컬 모델에 동시에 보내 "
            "답변을 나란히 보여줍니다."
        )

    if not st.session_state.app_settings["use_api"]:
        st.caption(
            "현재는 모델 없이 화면을 확인하는 "
            "테스트 모드입니다."
        )

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
            # 현재 사용자 세션에 질문 저장
            st.session_state.last_exchange = {
                "question": prompt,
                "answer": None,
                "dataset_name": dataset_name,
                "mode_name": mode_name,
            }

            with st.spinner(
                "요청을 처리하고 있어요. "
                "잠시만 기다려주세요...",
                show_time=True,
            ):
                try:
                    result = get_reply(
                        prompt,
                        DATASETS[dataset_name],
                        GENERATION_MODES[mode_name],
                    )

                except httpx.ConnectTimeout:
                    result = {
                        "error": (
                            "서버 연결 시간이 초과되었습니다. "
                            "잠시 후 다시 시도해주세요."
                        )
                    }

                except httpx.ReadTimeout:
                    result = {
                        "error": (
                            "API 응답 대기 시간이 초과되었습니다. "
                            "잠시 후 다시 시도해주세요."
                        )
                    }

                except httpx.TimeoutException:
                    result = {
                        "error": (
                            "요청 처리 시간이 초과되었습니다. "
                            "잠시 후 다시 시도해주세요."
                        )
                    }

                except httpx.HTTPStatusError as exc:
                    result = {
                        "error": (
                            "API 요청에 실패했습니다. "
                            f"응답 코드: {exc.response.status_code}"
                        )
                    }

                except httpx.RequestError:
                    result = {
                        "error": (
                            "FastAPI 서버에 연결하지 못했습니다. "
                            "API 주소와 서버 상태를 확인해주세요."
                        )
                    }

                except (ValueError, TypeError) as exc:
                    result = {
                        "error": str(exc)
                    }

            # 현재 사용자 세션에 응답 저장
            st.session_state.last_exchange["answer"] = result

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
            result_title = (
                "요약" if exchange["dataset_name"] == "문서" else "답변"
            )

            st.caption(
                f"🤖 {result_title} · 데이터셋: "
                f"{exchange['dataset_name']} · "
                f"{exchange.get('mode_name', 'OpenAI')}"
            )

            result = exchange["answer"]

            if isinstance(result, dict):
                if "error" in result:
                    st.error(result["error"])
                elif "results" in result:
                    show_comparison(result)
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
            "FastAPI 서버를 실행한 뒤 API 연결을 켜세요. "
            "모델이 연결되기 전에도 서버 통신을 테스트할 수 있어요."
        )

        check_connection = st.form_submit_button(
            "API 연결 확인"
        )

        submitted = st.form_submit_button(
            "저장하고 채팅으로 돌아가기"
        )

    # 입력한 주소로 서버 연결 확인
    if check_connection:
        test_url = api_url.strip().rstrip("/")

        if not validate_api_url(test_url):
            st.error(
                "유효한 http:// 또는 https:// "
                "API 주소를 입력해주세요."
            )

        else:
            try:
                with st.spinner(
                    "서버 연결을 확인하고 있어요.",
                    show_time=True,
                ):
                    health = check_api_connection(test_url)

                if health.get("mode") == "stub":
                    st.success(
                        "FastAPI 연결 성공 · "
                        "실제 모델은 아직 연결되지 않았습니다."
                    )
                else:
                    st.success(
                        "FastAPI 서버 연결 성공"
                    )

                st.caption(
                    "연결 확인 후 저장 버튼을 눌러 "
                    "설정을 적용하세요."
                )

            except httpx.TimeoutException:
                st.error(
                    "서버 연결 확인 시간이 초과되었습니다."
                )

            except httpx.HTTPStatusError as exc:
                st.error(
                    "서버 연결 확인 실패 · "
                    f"응답 코드: {exc.response.status_code}"
                )

            except httpx.RequestError:
                st.error(
                    "FastAPI 서버에 연결하지 못했습니다. "
                    "주소와 서버 실행 상태를 확인해주세요."
                )

            except (ValueError, TypeError):
                st.error(
                    "서버 상태 응답 형식이 올바르지 않습니다."
                )

    # 설정 저장
    if submitted:
        api_url = api_url.strip().rstrip("/")
        valid_url = validate_api_url(api_url)

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

initialize_session()

# 임시: FastAPI 연결 테스트
st.session_state.app_settings.update({
    "use_api": True,
    "api_url": "http://127.0.0.1:8000",
})


# ==========================================
# 10. 페이지 등록 및 라우팅
# ==========================================

chat_route = st.Page(
    chat_page,
    title="채팅",
    icon="💬",
    default=True,
)

settings_route = st.Page(
    settings_page,
    title="설정",
    icon="⚙️",
    url_path="settings",
)

tracing_route = st.Page(
    tracing_page.render,
    title="트레이싱",
    icon="📊",
    url_path="tracing",
)

# 페이지 이동은 사이드바에서 한다.
current_page = st.navigation(
    [chat_route, tracing_route, settings_route],
    position="sidebar",
)

# 조회 표가 넓게 보이도록 트레이싱 페이지만 넓은 레이아웃을 쓴다.
st.set_page_config(
    page_title="분류 챗봇",
    page_icon="💬",
    layout="wide" if current_page is tracing_route else "centered",
    initial_sidebar_state="expanded",
)

current_page.run()