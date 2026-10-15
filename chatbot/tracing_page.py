# -*- coding: utf-8 -*-
"""트레이싱 조회 페이지: 요청 목록, 호출 흐름, 통계를 보고 파일로 내보낸다.

FastAPI의 /admin API(fastapi/admin.py)를 admin_client.AdminClient로 부른다.
서버에 암호(TRACKING_ADMIN_PASSWORD)가 설정돼 있으면 이 페이지에서 입력해야 하고, 암호는 전송하지 않고 서명만 보낸다.
설정돼 있지 않으면 입력 없이 바로 열린다.
"""

import json
from datetime import date, datetime, time, timedelta, timezone

import altair as alt
import httpx
import pandas as pd
import streamlit as st

from admin_client import AdminAuthError, AdminClient

LABELS = ["announcement", "manual", "minutes", "shipping", "refund", "account"]
PRESETS = {"최근 1시간": timedelta(hours=1), "최근 24시간": timedelta(hours=24), "최근 7일": timedelta(days=7)}
SORTS = {"시작 시각": "started_at", "총 시간": "total_ms", "분류 시간": "classify_ms",
         "생성 시간": "generate_ms", "신뢰도": "confidence"}
SCOPES = {
    "요약 (요청 1건 = 1행)": "summary",
    "요청 (저장된 요청 기록 그대로)": "requests",
    "생성 호출별 (생성 모델 비교용, 호출 1번 = 1행)": "generations",
    "단계(span) (호출 단계 1개 = 1행)": "spans",
    "상세 (요청 + 호출 흐름)": "trace",
}
SCOPE_HELP = {
    "summary": "요청별 총 시간, 분류·생성 시간, 라벨, 신뢰도, 모델, 토큰 수를 한 줄로 모은 표입니다. 분석에 가장 편합니다.",
    "requests": "요청 기록 원본입니다. 상태코드, 오류, 입력 길이와 해시, 결과 요약이 들어 있습니다.",
    "generations": "생성 모델 호출 1번이 한 행입니다. /chat은 호출 하나, /compare는 제공자마다 한 행이라 같은 요청의 두 모델을 나란히 비교할 수 있습니다. 지연, 토큰, 토큰/초, 오류가 들어 있습니다.",
    "spans": "호출 단계(chat, compare, classify, apply_policy, generate, generate_<제공자>, respond)마다 한 행입니다. 단계별 지표는 attrs.* 열로 펼쳐집니다.",
    "trace": "요청과 그 호출 흐름을 함께 담습니다. json/jsonl은 요청 1건이 1개 객체(spans 포함)이고, csv는 span마다 요청 정보를 반복해서 한 줄씩 씁니다.",
}
FORMATS = {
    "csv": ("text/csv", "중첩된 값은 'a.b' 열로 펼쳐집니다. 한글이 Excel에서 깨지지 않도록 BOM을 붙입니다."),
    "json": ("application/json", "하나의 배열로 저장합니다. 중첩 구조가 그대로 유지됩니다."),
    "jsonl": ("application/x-ndjson", "한 줄에 객체 하나입니다. 큰 데이터를 줄 단위로 처리할 때 좋습니다."),
}
LOCAL_TZ = datetime.now().astimezone().tzinfo


# ==========================================
# 1. 공통 도우미
# ==========================================

def to_local(value):
    """저장된 UTC ISO 시각 -> 이 컴퓨터의 시간대 문자열"""
    if not value:
        return ""
    return datetime.fromisoformat(value).astimezone(LOCAL_TZ).strftime("%m-%d %H:%M:%S")


def ms_text(value):
    return "-" if value is None else f"{value:,.0f} ms"


def to_utc_iso(value):
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


@st.cache_data(ttl=10, show_spinner=False)
def fetch(api_url, password, kind, payload):
    """조회 결과를 10초 동안 재사용한다. (표 선택 같은 화면 갱신마다 서버를 다시 부르지 않기 위해)
    서명은 호출할 때마다 새로 만들어지므로 캐시 키에는 암호와 요청 내용만 들어간다."""
    client = AdminClient(api_url, password)
    args = json.loads(payload)

    if kind == "list":
        return client.list_requests(args["filters"], args["limit"], 0, args["sort"], args["direction"])

    if kind == "stats":
        return client.stats(args["filters"])

    if kind == "detail":
        return client.detail(args["id"])

    raise ValueError(kind)


def call(api_url, kind, **args):
    """fetch를 부르고 실패를 화면에 보여준다. 계속할 수 없으면 페이지 실행을 멈춘다."""
    password = st.session_state.get("admin_password") or ""
    payload = json.dumps(args, ensure_ascii=False, sort_keys=True)

    try:
        return fetch(api_url, password, kind, payload)

    except AdminAuthError:
        st.session_state.admin_ok = False
        st.session_state.admin_password = ""
        st.error("암호가 올바르지 않거나 서버의 암호가 바뀌었습니다. 다시 입력해주세요.")
        st.rerun()

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        detail = ""

        try:
            detail = exc.response.json().get("detail", "")
        except ValueError:
            pass

        if status == 404:
            return None

        st.error(f"조회에 실패했습니다 ({status}) {detail if isinstance(detail, str) else ''}")
        st.stop()

    except httpx.RequestError:
        st.error("FastAPI 서버에 연결하지 못했습니다. 서버 실행 상태와 설정의 API 주소를 확인해주세요.")
        st.stop()


# ==========================================
# 2. 암호 확인
# ==========================================

def ensure_access(api_url):
    """암호가 필요한 서버면 입력을 받는다. 통과하지 못하면 페이지 실행을 멈춘다."""
    try:
        info = AdminClient(api_url).auth_info()

    except httpx.RequestError:
        st.error("FastAPI 서버에 연결하지 못했습니다. 서버 실행 상태와 설정의 API 주소를 확인해주세요.")
        st.stop()

    except httpx.HTTPStatusError as exc:
        st.error(f"서버 응답 오류 ({exc.response.status_code}). 서버를 최신 버전으로 실행했는지 확인해주세요.")
        st.stop()

    if not info.get("store", True):
        st.warning("서버의 추적 저장소가 열려 있지 않아 조회할 데이터가 없습니다.")
        st.stop()

    if not info["required"]:
        st.session_state.admin_password = ""
        return False

    if st.session_state.get("admin_ok"):
        return True

    st.info("이 서버는 트레이싱 조회에 암호가 필요합니다. 암호는 서버로 보내지 않고, 서명만 보냅니다.")

    with st.form("admin_login"):
        password = st.text_input("암호", type="password")
        submitted = st.form_submit_button("확인")

    if submitted:
        try:
            AdminClient(api_url, password).check()

        except AdminAuthError:
            st.error("암호가 올바르지 않습니다.")

        except httpx.HTTPError:
            st.error("서버 확인에 실패했습니다.")

        else:
            st.session_state.admin_password = password
            st.session_state.admin_ok = True
            st.rerun()

    st.stop()


# ==========================================
# 3. 필터
# ==========================================

def filter_panel():
    """필터 위젯을 그리고 (필터 dict, 정렬, 방향, 최대 행 수)를 돌려준다."""
    with st.expander("🔎 필터", expanded=True):
        row1 = st.columns(4)
        preset = row1[0].selectbox("기간", [*PRESETS, "전체", "직접 지정"], index=1, key="f_preset")
        dataset = row1[1].selectbox("데이터셋", ["전체", "documents", "inquiries"], key="f_dataset")
        outcome = row1[2].selectbox("결과", ["전체", "success", "partial_success", "error", "cancelled"], key="f_outcome")
        label = row1[3].selectbox("분류 라벨", ["전체", *LABELS], key="f_label")

        since = until = None
        now = datetime.now(timezone.utc)

        if preset in PRESETS:
            since = to_utc_iso(now - PRESETS[preset])

        elif preset == "직접 지정":
            today = date.today()
            picked = st.date_input("날짜 범위", value=(today - timedelta(days=2), today), key="f_range")

            if len(picked) == 2:
                start = datetime.combine(picked[0], time.min, tzinfo=LOCAL_TZ)
                end = datetime.combine(picked[1] + timedelta(days=1), time.min, tzinfo=LOCAL_TZ)
                since, until = to_utc_iso(start), to_utc_iso(end)

        row2 = st.columns(4)
        status_code = row2[0].selectbox("HTTP 상태코드", ["전체", 200, 422, 500, 503], key="f_status")
        classification_status = row2[1].selectbox("분류 상태", ["전체", "success", "error", "timeout", "skipped"], key="f_cls")
        generation_status = row2[2].selectbox("생성 상태", ["전체", "success", "error", "timeout", "skipped"], key="f_gen")
        min_total = row2[3].number_input("총 시간 최소값 (ms, 0이면 제한 없음)", min_value=0, step=500, key="f_slow")

        row2b = st.columns(4)
        provider = row2b[0].selectbox("생성 모델", ["전체", "openai", "ollama"], key="f_provider",
                                      help="이 생성 모델을 호출한 요청. 비교(/compare) 요청은 어느 쪽이든 해당됩니다.")
        endpoint = row2b[1].selectbox("엔드포인트", ["전체", "/chat", "/compare"], key="f_endpoint")

        row3 = st.columns([2, 1, 1, 1])
        keyword = row3[0].text_input("입력 원문 키워드", key="f_q", placeholder="원문에 포함된 글자")
        sort_label = row3[1].selectbox("정렬", list(SORTS), key="f_sort")
        direction = row3[2].radio("방향", ["내림차순", "오름차순"], horizontal=True, key="f_dir")
        limit = row3[3].selectbox("목록 최대 행 수", [100, 200, 500, 1000], index=1, key="f_limit")

    def pick(value):
        return None if value == "전체" else value

    filters = {
        "since": since, "until": until, "dataset": pick(dataset), "outcome": pick(outcome), "label": pick(label),
        "status_code": pick(status_code), "classification_status": pick(classification_status),
        "generation_status": pick(generation_status), "min_total_ms": min_total or None, "q": keyword.strip() or None,
        "provider": pick(provider), "endpoint": pick(endpoint),
    }
    filters = {key: value for key, value in filters.items() if value is not None}

    return filters, SORTS[sort_label], "desc" if direction == "내림차순" else "asc", limit


# ==========================================
# 4. 개요 탭
# ==========================================

def latency_table(latency):
    names = {"total_ms": "총 시간", "classify_ms": "분류 (스팬)", "inference_ms": "분류 모델 계산",
             "non_inference_ms": "분류 대기·기타", "generate_ms": "생성 (OpenAI)"}
    rows = []

    for key, name in names.items():
        stat = latency.get(key)

        if stat:
            rows.append({"단계": name, "건수": stat["count"], "평균": stat["mean"], "p50": stat["p50"],
                         "p90": stat["p90"], "p95": stat["p95"], "p99": stat["p99"], "최대": stat["max"]})

    return pd.DataFrame(rows)


def provider_table(providers):
    rows = []

    for name, item in providers.items():
        latency, speed = item["latency_ms"] or {}, item["tokens_per_sec"] or {}
        rows.append({
            "생성 모델": name, "모델": ", ".join(item["models"]), "호출 수": item["count"],
            "성공률": item["success"] / item["count"] if item["count"] else None,
            "지연 p50(ms)": latency.get("p50"), "지연 p95(ms)": latency.get("p95"),
            "출력 토큰 평균": (item["output_tokens"] or {}).get("mean"), "토큰/초 p50": speed.get("p50"),
            "입력 토큰 합": item["tokens"]["input"], "출력 토큰 합": item["tokens"]["output"],
            "콜드 스타트": item["cold_starts"], "문맥 초과": item["context_full"],
        })

    return pd.DataFrame(rows)


def provider_section(stats):
    """생성 모델(OpenAI, 로컬 Ollama)별 비교. 지연·토큰 분포는 성공한 호출만으로 계산한 값이다."""
    providers = stats.get("providers") or {}

    if not providers:
        return

    st.markdown("**생성 모델 비교**")
    table = provider_table(providers)
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "성공률": st.column_config.NumberColumn(format="percent"),
        "지연 p50(ms)": st.column_config.NumberColumn(format="%.0f"),
        "지연 p95(ms)": st.column_config.NumberColumn(format="%.0f"),
        "출력 토큰 평균": st.column_config.NumberColumn(format="%.0f"),
        "토큰/초 p50": st.column_config.NumberColumn(format="%.1f"),
    })

    chart = table.dropna(subset=["지연 p50(ms)"])

    if not chart.empty and len(chart) > 1:
        left, right = st.columns(2)

        with left:
            st.caption("지연 p50 / p95 (ms, 성공한 호출)")
            data = chart.melt(id_vars="생성 모델", value_vars=["지연 p50(ms)", "지연 p95(ms)"], var_name="구분", value_name="ms")
            st.altair_chart(
                alt.Chart(data).mark_bar().encode(
                    x=alt.X("생성 모델:N", title=None), xOffset="구분:N", y=alt.Y("ms:Q", title=None),
                    color=alt.Color("구분:N", legend=alt.Legend(orient="top", title=None)),
                    tooltip=["생성 모델", "구분", alt.Tooltip("ms:Q", format=",.0f")]),
                width="stretch")

        with right:
            st.caption("성공률")
            st.altair_chart(
                alt.Chart(table).mark_bar().encode(
                    x=alt.X("생성 모델:N", title=None), y=alt.Y("성공률:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title=None),
                    tooltip=["생성 모델", alt.Tooltip("성공률:Q", format=".1%")]),
                width="stretch")

    comparison = stats.get("comparison") or {}

    if comparison.get("requests"):
        faster = ", ".join(f"{name} {count}건" for name, count in comparison["faster"].items()) or "-"
        st.caption(f"비교(/compare) 요청 {comparison['requests']}건 중 모든 모델이 성공한 요청 {comparison['all_success']}건 · "
                   f"더 빨랐던 쪽: {faster}")

    failures = {name: item["errors"] for name, item in providers.items() if item["errors"]}

    for name, errors in failures.items():
        with st.expander(f"{name} 오류 ({sum(errors.values())}건)"):
            st.dataframe(pd.DataFrame({"사유": list(errors), "건수": list(errors.values())}), hide_index=True, width="stretch")


def overview_tab(stats):
    if not stats["total"]:
        st.info("조건에 맞는 요청이 없습니다.")
        return

    outcomes = stats["outcomes"]
    total = stats["total"]
    success = outcomes.get("success", 0)
    total_latency = stats["latency"].get("total_ms") or {}
    tokens = stats["tokens"]

    cols = st.columns(5)
    cols[0].metric("요청 수", f"{total:,}")
    cols[1].metric("성공률", f"{success / total:.1%}")
    cols[2].metric("실패·부분 실패", f"{total - success:,}건")
    cols[3].metric("총 시간 p50", ms_text(total_latency.get("p50")))
    cols[4].metric("총 시간 p95", ms_text(total_latency.get("p95")))

    if stats["truncated_rows"]:
        st.warning("집계 한도(100,000건)를 넘어 일부 요청만 반영됐습니다. 기간을 좁혀보세요.")

    timeline = pd.DataFrame(stats["timeline"])

    if not timeline.empty:
        timeline["시각"] = pd.to_datetime(timeline["bucket"], utc=True).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        unit = "시간별" if stats["bucket"] == "hour" else "일별"
        left, right = st.columns(2)

        with left:
            st.markdown(f"**{unit} 요청 수**")
            data = timeline.assign(성공=timeline["count"] - timeline["errors"], 실패=timeline["errors"])
            data = data.melt(id_vars="시각", value_vars=["성공", "실패"], var_name="구분", value_name="건수")
            st.altair_chart(
                alt.Chart(data).mark_bar().encode(
                    x=alt.X("시각:T", title=None), y=alt.Y("건수:Q", title=None),
                    color=alt.Color("구분:N", scale=alt.Scale(domain=["성공", "실패"], range=["#4c78a8", "#e45756"]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=["시각:T", "구분:N", "건수:Q"]),
                width="stretch")

        with right:
            st.markdown(f"**{unit} 평균 총 시간 (ms)**")
            st.altair_chart(
                alt.Chart(timeline).mark_line(point=True).encode(
                    x=alt.X("시각:T", title=None), y=alt.Y("avg_total_ms:Q", title=None),
                    tooltip=["시각:T", alt.Tooltip("avg_total_ms:Q", format=",.0f", title="평균 ms")]),
                width="stretch")

    left, mid, right = st.columns(3)

    with left:
        st.markdown("**결과 분포**")
        st.bar_chart(pd.Series(outcomes, name="건수"))

    with mid:
        st.markdown("**데이터셋별 요청**")
        st.bar_chart(pd.Series(stats["datasets"], name="건수"))

    with right:
        st.markdown("**HTTP 상태코드**")
        st.bar_chart(pd.Series(stats["status_codes"], name="건수"))

    if stats["labels"]:
        st.markdown("**분류 라벨 분포**")
        columns = st.columns(len(stats["labels"]))

        for column, (dataset, counts) in zip(columns, stats["labels"].items()):
            with column:
                st.caption(dataset)
                st.bar_chart(pd.Series(counts, name="건수"))

    st.markdown("**단계별 지연 분포 (ms)**")
    table = latency_table(stats["latency"])

    if not table.empty:
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            name: st.column_config.NumberColumn(format="%.0f") for name in ["평균", "p50", "p90", "p95", "p99", "최대"]})

    provider_section(stats)

    note = []
    confidence = stats.get("confidence")

    if confidence:
        note.append(f"분류 신뢰도 평균 {confidence['mean']:.3f} (최소 {confidence['min']:.3f})")

    note.append(f"입력이 잘린 요청 {stats['truncated_inputs']}건")
    note.append(f"LLM 토큰 {tokens['llm_input'] + tokens['llm_output']:,} (입력 {tokens['llm_input']:,} · 출력 {tokens['llm_output']:,})")
    st.caption(" · ".join(note))

    if stats["errors"]:
        st.markdown("**오류 / 실패 사유**")
        st.dataframe(pd.DataFrame({"사유": list(stats["errors"]), "건수": list(stats["errors"].values())}),
                     hide_index=True, width="stretch")


# ==========================================
# 5. 목록 탭
# ==========================================

def provider_text(item):
    """목록에 보여줄 생성 모델. /compare는 여러 모델이라 '비교(openai+ollama)'로 표시한다."""
    if item.get("llm_provider"):
        return item["llm_provider"]

    if item.get("llm_providers"):
        try:
            return "비교(" + "+".join(json.loads(item["llm_providers"])) + ")"
        except ValueError:
            return "비교"

    return None


def list_dataframe(items):
    rows = []

    for item in items:
        rows.append({
            "시작": to_local(item["started_at"]),
            "데이터셋": item["dataset"],
            "결과": item["outcome"],
            "생성 모델": provider_text(item),
            "상태": item["status_code"],
            "총 시간(ms)": item["total_ms"],
            "분류(ms)": item["classify_ms"],
            "생성(ms)": item["generate_ms"],
            "라벨": item["label"],
            "신뢰도": item["confidence"],
            "분류 모델": item["classifier_model"],
            "LLM 입력 토큰": item["llm_input_tokens"],
            "LLM 출력 토큰": item["llm_output_tokens"],
            "오류": item["error"],
            "입력 미리보기": item["input_preview"],
            "request_id": item["request_id"],
        })

    return pd.DataFrame(rows)


def list_tab(api_url, filters, sort, direction, limit):
    result = call(api_url, "list", filters=filters, limit=limit, sort=sort, direction=direction)
    items = result["items"]
    st.session_state.list_total = result["total"]
    st.session_state.list_items = items

    if not items:
        st.info("조건에 맞는 요청이 없습니다.")
        return

    st.caption(f"조건에 맞는 요청 {result['total']:,}건 중 {len(items):,}건 표시 · 행을 선택하면 상세 보기와 내보내기에 쓸 수 있습니다.")
    frame = list_dataframe(items)

    event = st.dataframe(
        frame, hide_index=True, width="stretch", height=520, key="req_table",
        on_select="rerun", selection_mode="multi-row",
        column_config={
            "총 시간(ms)": st.column_config.NumberColumn(format="%.0f"),
            "분류(ms)": st.column_config.NumberColumn(format="%.0f"),
            "생성(ms)": st.column_config.NumberColumn(format="%.0f"),
            "신뢰도": st.column_config.NumberColumn(format="%.3f"),
            "입력 미리보기": st.column_config.TextColumn(width="large"),
            "request_id": st.column_config.TextColumn(width="small"),
        })

    rows = event.selection.rows if event and event.selection else []
    st.session_state.selected_ids = [items[index]["request_id"] for index in rows if index < len(items)]


# ==========================================
# 6. 상세 탭
# ==========================================

def span_offsets(spans):
    """각 span의 시작 위치(ms)를 최상위 span 시작 기준으로 계산한다."""
    origin = datetime.fromisoformat(spans[0]["started_at"])
    rows = []

    for span in spans:
        start = (datetime.fromisoformat(span["started_at"]) - origin).total_seconds() * 1000
        duration = span["duration_ms"] or 0
        rows.append({"seq": span["seq"], "단계": span["name"], "시작(ms)": start, "끝(ms)": start + duration,
                     "소요(ms)": duration, "상태": span["status"]})

    return pd.DataFrame(rows)


def answer_block(generation, key):
    """답변과 다음 단계를 보여준다. 원문 저장이 꺼져 있었다면 답변은 비어 있다."""
    if generation.get("answer"):
        st.text_area("답변", generation["answer"], height=140, disabled=True, label_visibility="collapsed", key=f"answer_{key}")

        for step in generation.get("next_steps") or []:
            st.markdown(f"- {step}")

    if generation.get("error"):
        st.warning(generation["error"])


def comparison_block(generations, spans):
    """/compare 요청: 같은 프롬프트에 대한 생성 모델별 답변과 지표를 나란히 보여준다."""
    st.markdown("**생성 모델 비교**")
    metrics = {span["name"].removeprefix("generate_"): span["attrs"] for span in spans if span["name"].startswith("generate_")}
    columns = st.columns(len(generations))

    for column, generation in zip(columns, generations):
        provider = generation["provider"]
        attrs = metrics.get(provider, {})

        with column:
            st.markdown(f"##### {provider} · {generation['status']}")
            facts = [f"모델 {attrs['model']}"] if attrs.get("model") else []

            if attrs.get("latency_ms") is not None:
                facts.append(f"지연 {attrs['latency_ms']:,.0f} ms")

            if attrs.get("input_tokens") is not None:
                facts.append(f"토큰 입력 {attrs['input_tokens']:,} / 출력 {attrs.get('output_tokens') or 0:,}")

            if attrs.get("tokens_per_sec"):
                facts.append(f"{attrs['tokens_per_sec']:.1f} 토큰/초")

            if attrs.get("load_ms") and attrs["load_ms"] > 1000:
                facts.append(f"모델 로딩 {attrs['load_ms'] / 1000:.1f}초 (콜드 스타트)")

            if attrs.get("context_full"):
                facts.append("⚠️ 문맥 길이 한도에 도달")

            st.caption(" · ".join(facts))
            answer_block(generation, provider)


def detail_tab(api_url):
    selected = st.session_state.get("selected_ids", [])
    request_id = None

    left, right = st.columns([2, 3])

    with left:
        if selected:
            request_id = st.selectbox("목록에서 선택한 요청", selected, format_func=lambda value: value[:12] + "…")

        typed = st.text_input("또는 요청 ID 직접 입력", placeholder="request_id 또는 Streamlit이 보낸 client_request_id")

        if typed.strip():
            request_id = typed.strip()

    if not request_id:
        st.info("‘요청 목록’에서 행을 선택하거나 요청 ID를 입력하면 호출 흐름을 볼 수 있습니다.")
        return

    detail = call(api_url, "detail", id=request_id)

    if detail is None:
        st.warning("해당 요청을 찾을 수 없습니다.")
        return

    request, spans = detail["request"], detail["spans"]

    cols = st.columns(5)
    cols[0].metric("결과", request["outcome"])
    cols[1].metric("HTTP 상태", request["status_code"])
    cols[2].metric("총 시간", ms_text(request["duration_ms"]))
    cols[3].metric("데이터셋", request["dataset"] or "-")
    cols[4].metric("시작", to_local(request["started_at"]))
    st.caption(f"request_id: `{request['request_id']}` · client_request_id: `{request['client_request_id'] or '-'}`"
               f" · {request['method']} {request['path']}")

    if request["error_type"] or request["error_message"]:
        st.error(f"{request['error_type']}: {request['error_message']}")

    if request["error_detail"]:
        st.code(request["error_detail"], language="json")

    texts = st.columns(2)

    with texts[0]:
        st.markdown("**입력 원문**")

        if request["input_text"] is not None:
            st.text_area("입력", request["input_text"], height=220, disabled=True, label_visibility="collapsed")
        else:
            st.caption(f"원문이 저장되지 않았습니다. (길이 {request['input_len']}, 해시 {request['input_sha256']})")

    with texts[1]:
        st.markdown("**결과**")
        output = request["output"]

        if output:
            classification = output["classification"]
            st.write(f"분류: **{classification['label']}** (신뢰도 {classification['confidence']}) · 상태 {classification['status']}")

            if classification.get("error"):
                st.warning(classification["error"])

            if "generation" in output:
                generation = output["generation"]
                st.write(f"생성 상태: {generation['status']}" + (f" · {generation['provider']}" if generation.get("provider") else ""))
                answer_block(generation, "single")
        else:
            st.caption("결과 기록이 없습니다. (검증 실패나 서버 오류)")

    if output and "generations" in output:
        comparison_block(output["generations"], spans)

    if not spans:
        st.info("이 요청에는 호출 흐름(span) 기록이 없습니다.")
        return

    st.markdown("**호출 흐름**")
    offsets = span_offsets(spans)
    order = list(offsets.sort_values("seq")["단계"])
    colors = alt.Scale(domain=["success", "partial_success", "skipped", "timeout", "error"],
                       range=["#4c78a8", "#f2a541", "#b0b0b0", "#e45756", "#e45756"])
    st.altair_chart(
        alt.Chart(offsets).mark_bar(size=18).encode(
            y=alt.Y("단계:N", sort=order, title=None),
            x=alt.X("시작(ms):Q", title="시작 이후 경과 (ms)"), x2="끝(ms):Q",
            color=alt.Color("상태:N", scale=colors, legend=alt.Legend(orient="top", title=None)),
            tooltip=["단계", alt.Tooltip("소요(ms):Q", format=",.1f"), "상태"]),
        width="stretch")

    by_id = {span["run_id"]: span["name"] for span in spans}
    table = pd.DataFrame([{
        "순서": span["seq"], "단계": span["name"], "상위": by_id.get(span["parent_run_id"], "-"),
        "소요(ms)": span["duration_ms"], "상태": span["status"], "오류": span["error_type"] or "",
    } for span in spans])
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"소요(ms)": st.column_config.NumberColumn(format="%.1f")})

    for span in spans:
        if span["attrs"]:
            with st.expander(f"{span['name']} 지표 ({len(span['attrs'])}개)"):
                st.json(span["attrs"])

    with st.expander("원본 JSON"):
        st.json(detail)


# ==========================================
# 7. 내보내기 탭
# ==========================================

def export_tab(api_url, filters):
    selected = st.session_state.get("selected_ids", [])
    total = st.session_state.get("list_total", 0)

    left, right = st.columns(2)

    with left:
        scope_label = st.radio("내보낼 내용", list(SCOPES), key="x_scope")
        scope = SCOPES[scope_label]
        st.caption(SCOPE_HELP[scope])

    with right:
        file_format = st.radio("파일 형식", list(FORMATS), horizontal=True, key="x_format")
        st.caption(FORMATS[file_format][1])

        targets = [f"현재 필터에 맞는 전체 ({total:,}건)"]

        if selected:
            targets.append(f"목록에서 선택한 요청 ({len(selected)}건)")

        target = st.radio("대상", targets, key="x_target")
        include_text = st.checkbox("입력 원문과 답변 포함", key="x_text")

        if include_text:
            st.warning("원문과 답변에는 개인정보가 들어 있을 수 있습니다. 파일을 안전하게 다뤄주세요.")

    use_selected = target.startswith("목록에서")
    limit = st.number_input("최대 행 수", min_value=1, max_value=100000, value=100000, step=1000, key="x_limit")
    body = {"format": file_format, "scope": scope, "include_text": include_text, "limit": int(limit)}

    if use_selected:
        body["request_ids"] = selected
    else:
        body["filters"] = filters

    if st.button("내보내기 파일 만들기", type="primary"):
        password = st.session_state.get("admin_password") or ""

        try:
            with st.spinner("파일을 만들고 있어요."):
                name, data, rows = AdminClient(api_url, password).export(body)

        except AdminAuthError:
            st.session_state.admin_ok = False
            st.error("암호가 올바르지 않습니다. 다시 입력해주세요.")
            st.rerun()

        except httpx.HTTPStatusError as exc:
            st.error(f"내보내기에 실패했습니다 ({exc.response.status_code}).")

        except httpx.RequestError:
            st.error("FastAPI 서버에 연결하지 못했습니다.")

        else:
            st.session_state.export_result = {"name": name, "data": data, "rows": rows, "format": file_format}

    result = st.session_state.get("export_result")

    if result:
        st.success(f"{result['rows']:,}행, {len(result['data']) / 1024:,.1f} KB 파일이 준비됐어요.")
        st.download_button(f"⬇️ {result['name']} 다운로드", result["data"], file_name=result["name"],
                           mime=FORMATS[result["format"]][0])


# ==========================================
# 8. 페이지
# ==========================================

def render():
    api_url = st.session_state.app_settings["api_url"]

    title, refresh = st.columns([6, 1])
    title.title("📊 트레이싱")

    if refresh.button("🔄 새로고침"):
        st.cache_data.clear()
        st.rerun()

    protected = ensure_access(api_url)

    if protected and st.sidebar.button("🔒 암호 지우고 잠그기"):
        st.session_state.admin_ok = False
        st.session_state.admin_password = ""
        st.rerun()

    filters, sort, direction, limit = filter_panel()
    stats = call(api_url, "stats", filters=filters)

    overview, listing, detail, export = st.tabs(["📈 개요", "📋 요청 목록", "🔍 요청 상세", "📤 내보내기"])

    with overview:
        overview_tab(stats)

    with listing:
        list_tab(api_url, filters, sort, direction, limit)

    with detail:
        detail_tab(api_url)

    with export:
        export_tab(api_url, filters)
