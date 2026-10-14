"""
이 컴퓨터에서 CLM을 구동하거나 파인튜닝할 수 있는 상태인지 점검한다.

점검 결과는 [OK] / [경고] / [실패] 로 보여 주고, 경고와 실패에는 고치는 방법을 함께 적는다.
  1. 컴퓨터        : 운영체제, 메모리, GPU
  2. 파이썬 환경   : 파이썬 버전, 필수 패키지
  3. 외부 도구     : CLM 저장소, llama-server, GPU 인식
  4. 모델 파일     : GGUF, 공식 헤드, 토크나이저, 저장된 헤드(best_model)와 인코더의 짝
  5. 파인튜닝 준비 : 데이터 파일, 입력 길이, 저장 공간, 헤드 학습 동작
  6. 인코더 서버   : 포트와 VRAM, 서버를 실제로 띄워 기준 벡터와 비교, 속도와 예상 임베딩 시간 (--static이면 생략)

점검은 아무것도 내려받지 않는다. 모델 파일이 없으면 [경고]로 알리고 서버 확인은 건너뛴다. (처음 실행하면 코드가 자동으로 내려받는다)
프로젝트 파일도 바꾸지 않는다. 임시 파일을 쓰고 지우는 것과, 서버가 없을 때 임시로 띄웠다가 끄는 것만 한다. (서버를 띄우면 jev/tmp에 서버 로그가 남는다)
이 파일은 표준 라이브러리만으로 시작하므로 패키지가 빠진 컴퓨터에서도 무엇이 빠졌는지 알려 준다.

루트 디렉토리 기준 실행 코드 (Windows는 PYTHONUTF8=1 권장)
uv run python ./jev/scripts/step05_check_clm_env.py                # 전체 점검 (인코더 서버를 실제로 확인, 1분 안팎)
uv run python ./jev/scripts/step05_check_clm_env.py --static       # 서버를 띄우지 않고 설치와 파일 상태만 (GPU를 다른 작업이 쓰는 중일 때)
uv run python ./jev/scripts/step05_check_clm_env.py --for run      # 구동만 필요할 때 (파인튜닝 준비 항목의 실패는 종료 코드에 반영하지 않는다)
종료 코드: 0 = 해당 용도로 쓸 수 있음, 1 = 실패 항목이 있음
"""

import argparse
import contextlib
import ctypes
import importlib
import io
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent.parent
OK, WARN, FAIL, SKIP, INFO = "OK", "경고", "실패", "건너뜀", "정보"

GPU_LAYERS_FULL = 36          # Qwen3-8B의 레이어 수
MIN_FREE_RAM_GB = 6
MIN_FREE_DISK_GB = 2          # artifacts에 쌓이는 결과(데이터셋 10개에 약 1GB)와 임베딩 캐시를 둘 공간
DOWNLOAD_GB = {"GGUF": 5.1, "공식 헤드": 0.1, "토크나이저": 0.05}
DEFAULT_LONG_TOKENS = 1300    # 데이터에서 가장 긴 글(약 1,230토큰)보다 조금 긴 값. 데이터를 못 읽었을 때만 쓴다.
PACKAGES = (("numpy", "numpy"), ("requests", "requests"), ("torch", "torch"), ("transformers", "transformers"),
            ("safetensors", "safetensors"), ("huggingface_hub", "huggingface-hub"), ("dotenv", "python-dotenv"),
            ("sklearn", "scikit-learn"))

results = []     # (용도 run/train/info, 상태, 이름)
ctx = {"downloads": []}   # 점검끼리 주고받는 값. 키가 있어야 그 값을 쓰는 점검을 실행한다.
pending_skips = []        # 앞 단계가 준비되지 않아 건너뛴 점검의 이름. 한 줄로 모아서 알린다.


# ═════════════════════════ 출력 · 실행 틀 ═════════════════════════
def flush_skips() -> None:
    if pending_skips:
        print(f"  [{SKIP}] 앞 단계가 준비되지 않아 건너뛴 점검: {', '.join(pending_skips)}")
        pending_skips.clear()


def section(title: str) -> None:
    flush_skips()
    print(f"\n[{title}]")


def report(scope: str, status: str, name: str, detail: str = "", hint: str = "") -> None:
    results.append((scope, status, name))
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    if hint and status in (WARN, FAIL):
        for number, line in enumerate(str(hint).splitlines()):
            print("        " + ("-> " if number == 0 else "   ") + line)


def run_check(scope: str, name: str, func, needs=()):
    """func() -> (상태, 설명, 고치는 방법). 예외가 나도 그 점검만 실패로 표시하고 다음으로 넘어간다."""
    if [key for key in needs if key not in ctx]:
        results.append((scope, SKIP, name))
        pending_skips.append(name)
        return None
    try:
        status, detail, hint = func()
    except Exception as error:
        lines = str(error).splitlines() or [""]
        status, detail, hint = FAIL, f"{type(error).__name__}: {lines[0]}", "\n".join(lines[1:])
    report(scope, status, name, detail, hint)
    return status


def run_command(args: list, cwd=None, timeout: float = 60) -> tuple[int, str]:
    """(종료 코드, stdout + stderr). 실행하지 못하면 (-1, 오류 문장)."""
    try:
        done = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              cwd=cwd, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        return -1, f"{type(error).__name__}: {error}"
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def nearest_existing(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


# ═════════════════════════ 1. 컴퓨터 ═════════════════════════
def memory_gb() -> tuple[float, float] | None:
    """(전체, 지금 쓸 수 있는) 메모리 GB. 알 수 없으면 None."""
    if sys.platform == "win32":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                        ("page_avail", ctypes.c_ulonglong), ("virt_total", ctypes.c_ulonglong),
                        ("virt_avail", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.total / 2**30, status.avail / 2**30
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        return os.sysconf("SC_PHYS_PAGES") * page / 2**30, os.sysconf("SC_AVPHYS_PAGES") * page / 2**30
    except (ValueError, OSError, AttributeError):
        return None


def nvidia_smi() -> tuple[str, str] | None:
    """(GPU 이름, 드라이버 버전). nvidia-smi가 없으면 None."""
    code, text = run_command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i", "0"], timeout=15)
    if code != 0 or "," not in text:
        return None
    name, driver = [part.strip() for part in text.strip().splitlines()[0].split(",", 1)]
    return name, driver


def check_machine():
    ram = memory_gb()
    gpu = nvidia_smi()
    parts = [f"{platform.system()} {platform.release()}", f"CPU {os.cpu_count()}코어"]
    parts.append(f"메모리 {ram[0]:.0f}GB 중 {ram[1]:.1f}GB 사용 가능" if ram else "메모리 확인 불가")
    parts.append(f"GPU {gpu[0]} (드라이버 {gpu[1]})" if gpu else "nvidia-smi 없음")
    if ram and ram[1] < MIN_FREE_RAM_GB:
        return WARN, " | ".join(parts), (f"사용 가능한 메모리가 {MIN_FREE_RAM_GB}GB보다 적으면 8GB GGUF 파일을 올리는 동안 느려지거나 실패할 수 있습니다. "
                                       "다른 프로그램을 닫으세요.")
    return OK, " | ".join(parts), ""


# ═════════════════════════ 2. 파이썬 환경 ═════════════════════════
def check_python():
    problems = []
    if sys.version_info[:2] != (3, 12):
        problems.append("프로젝트는 Python 3.12 기준입니다(pyproject.toml).")
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        problems.append("프로젝트의 .venv에서 실행한 것이 아닙니다.")
    detail = f"{platform.python_version()} ({sys.executable})"
    if problems:
        return WARN, detail, " ".join(problems) + "\n`uv sync` 후 `uv run python ...`으로 실행하세요."
    return OK, detail, ""


def check_packages():
    versions, missing = [], []
    for module, dist in PACKAGES:
        try:
            importlib.import_module(module)
        except Exception as error:               # 설치되지 않았거나, 있어도 불러오다 실패한 경우
            missing.append(f"{dist}({type(error).__name__})")
            continue
        try:
            versions.append(f"{dist} {metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            versions.append(dist)
    if missing:
        return FAIL, "없거나 불러올 수 없는 패키지: " + ", ".join(missing), "`uv sync`로 설치하세요."
    ctx["packages"] = True
    return OK, ", ".join(versions), ""


def check_torch():
    import torch
    cuda = torch.cuda.is_available()
    wanted = ctx["S"].CLM_CONFIG["head_device"]
    detail = f"CUDA {'사용 가능' if cuda else '없음'} | 헤드 학습 장치 설정: {wanted}"
    if wanted.startswith("cuda") and not cuda:
        return FAIL, detail, "clm_settings.py의 CLM_CONFIG['head_device']가 cuda인데 torch가 GPU를 쓸 수 없습니다. cpu로 바꾸세요."
    return INFO, detail, ""


def load_project():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import clm_settings as S
    from util import clm_encoder as E
    from util import clm_head as H
    ctx.update(S=S, E=E, H=H, modules=True)
    return OK, "clm_settings, util.clm_encoder, util.clm_head", ""


# ═════════════════════════ 3. 외부 도구 ═════════════════════════
def check_clm_repo():
    S, E = ctx["S"], ctx["E"]
    with contextlib.redirect_stdout(io.StringIO()):          # 커밋이 다르다는 안내는 아래에서 직접 알린다
        try:
            clm = E.load_clm()
        except FileNotFoundError:
            return FAIL, f"CLM 저장소가 없습니다: {S.CLM_REPO}", (
                f"git clone https://github.com/Contrastive-LM/CLM.git \"{S.CLM_REPO}\"\n"
                f"git -C \"{S.CLM_REPO}\" checkout {S.CLM_REPO_COMMIT}\n.env의 CLM_REPO로 다른 위치를 지정할 수도 있습니다.")
    ctx["clm"] = True
    if clm.commit is None:
        return WARN, f"{S.CLM_REPO} (git이 없어 커밋을 확인하지 못했습니다)", ""
    if clm.commit != S.CLM_REPO_COMMIT:
        return WARN, f"{S.CLM_REPO} (커밋 {clm.commit[:7]})", (
            f"검증한 커밋은 {S.CLM_REPO_COMMIT[:7]}입니다. 학습 방식이 달라졌을 수 있으니 "
            f"git -C \"{S.CLM_REPO}\" checkout {S.CLM_REPO_COMMIT} 로 맞추세요.")
    return OK, f"{S.CLM_REPO} (커밋 {clm.commit[:7]})", ""


def check_llama_server():
    S, E = ctx["S"], ctx["E"]
    server = E.find_llama_server()
    if server is None:
        return FAIL, "llama-server를 찾지 못했습니다", (
            f"llama.cpp Windows CUDA 빌드(llama-bXXXXX-bin-win-cuda-12.4-x64.zip과 cudart-llama-bin-win-cuda-12.4-x64.zip)를 "
            f"같은 폴더에 풀어 {S.TOOLS_DIR / 'llama.cpp'} 에 두거나\n.env의 CLM_LLAMA_SERVER에 llama-server 경로를 적으세요. "
            "(https://github.com/ggml-org/llama.cpp/releases)")
    code, text = run_command([str(server), "--version"], cwd=server.parent)
    build = re.search(r"build (\d+)", text)
    if code != 0 or not build:
        return FAIL, f"{server} 를 실행하지 못했습니다", (
            "필요한 DLL(cudart, cublas)이 llama-server.exe와 같은 폴더에 있는지 확인하세요.\n" + text.strip()[-300:])
    ctx["server"] = server
    return OK, f"{server} (build {build.group(1)})", ""


def check_gpu_devices():
    server = ctx["server"]
    _, text = run_command([str(server), "--list-devices"], cwd=server.parent, timeout=120)
    devices = [(name, desc.strip(), int(total), int(free)) for name, desc, total, free in
               re.findall(r"^\s*([A-Za-z]+\d+):\s*(.+?)\s*\((\d+) MiB,\s*(\d+) MiB free\)", text, re.MULTILINE)]
    if not devices:
        return FAIL, "llama.cpp가 GPU를 찾지 못했습니다 (CPU만으로는 8B 인코더가 너무 느려서 쓸 수 없습니다)", (
            "NVIDIA 드라이버를 최신으로 업데이트하고 CUDA 12.4 빌드를 쓰세요. NVIDIA GPU가 아니라면 Vulkan 빌드를 쓸 수 있습니다.\n"
            + (text.strip().splitlines() or [""])[-1][:200])
    ctx["devices"] = devices
    name, desc, total, free = devices[0]
    return OK, f"{name}: {desc} (VRAM {total}MiB, 지금 여유 {free}MiB)", ""


# ═════════════════════════ 4. 모델 파일 ═════════════════════════
def local_hf_file(repo: str, filename: str) -> Path | None:
    """HF 캐시에 이미 받아 둔 파일의 경로. 없으면 None. (내려받지 않는다)"""
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(repo, filename, local_files_only=True))
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_gguf():
    S = ctx["S"]
    enc = S.ENCODER
    path = local_hf_file(enc["gguf_repo"], enc["gguf_file"])
    if path is None:
        from huggingface_hub import constants
        partial = list((Path(constants.HF_HUB_CACHE) / ("models--" + enc["gguf_repo"].replace("/", "--")) / "blobs").glob("*.incomplete"))
        ctx["downloads"].append("GGUF")
        note = " (내려받다 만 파일이 있어 처음부터 다시 받습니다)" if partial else ""
        return WARN, f"{enc['gguf_file']}이 HF 캐시에 없습니다{note}", (
            "처음 실행할 때 코드가 자동으로 내려받습니다 (약 5GB). 점검에서는 내려받지 않으므로 서버 확인은 건너뜁니다.")
    with path.open("rb") as stream:
        magic = stream.read(4)
    if magic != b"GGUF":
        return FAIL, f"{path} 가 GGUF 파일이 아닙니다 (손상됐을 수 있습니다)", "HF 캐시에서 이 파일을 지우면 다음 실행 때 다시 내려받습니다."
    ctx["gguf"] = path
    return OK, f"{enc['gguf_file']} ({path.stat().st_size / 2**30:.2f}GiB)", ""


def check_base_head():
    S = ctx["S"]
    path = local_hf_file(S.BASE_HEAD["repo"], S.BASE_HEAD["file"])
    if path is None:
        ctx["downloads"].append("공식 헤드")
        return WARN, f"{S.BASE_HEAD['file']}이 HF 캐시에 없습니다", "처음 실행할 때 코드가 자동으로 내려받습니다 (약 72MB)."
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)     # 파일 안의 코드가 실행되지 않는 방식
    needed = {"state_head", "action_head", "logit_scale", "cfg"}
    if not needed <= set(checkpoint):
        return FAIL, f"{path.name}에 {sorted(needed - set(checkpoint))} 항목이 없습니다", "HF 캐시에서 이 파일을 지우면 다시 내려받습니다."
    ctx["head"] = path
    cfg = checkpoint["cfg"]
    return OK, f"{path.name} ({path.stat().st_size / 2**20:.0f}MiB, {cfg['hidden_size']} -> {cfg['width']} x{cfg['depth']} -> {checkpoint['projection_dim']})", ""


def check_tokenizer():
    S = ctx["S"]
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(S.ENCODER["tokenizer"], local_files_only=True)
    except Exception:
        ctx["downloads"].append("토크나이저")
        return WARN, f"{S.ENCODER['tokenizer']} 토크나이저가 HF 캐시에 없습니다", "처음 실행할 때 코드가 자동으로 내려받습니다 (약 11MB)."
    ctx["tokenizer"] = True
    return OK, f"{S.ENCODER['tokenizer']} ({len(tok)}개 토큰)", ""


def check_saved_heads():
    S, H = ctx["S"], ctx["H"]
    files = sorted((ROOT / "artifacts").glob("*/*/clm/best_model/encoder.json"))
    if not files:
        return INFO, "저장된 CLM 헤드가 아직 없습니다 (파인튜닝 전이면 정상입니다)", ""
    gguf = ctx.get("gguf")
    current = {"gguf_file": S.ENCODER["gguf_file"], "pooling": S.ENCODER["pooling"], "max_len": S.ENCODER["ctx"],
               "hidden_size": 4096, "tokenizer": S.ENCODER["tokenizer"],
               "gguf_size_bytes": gguf.stat().st_size if gguf else None}
    base_hash = file_sha256(ctx["head"]) if "head" in ctx else None
    broken, other_base = [], 0
    for file in files:
        saved = json.loads(file.read_text(encoding="utf-8"))
        where = "/".join(file.parts[-5:-3])
        try:
            H.check_same_encoder(saved, current)
        except ValueError as error:
            broken.append(f"{where}: " + str(error).splitlines()[1].strip())
        recorded = (saved.get("base_head") or {}).get("sha256")
        if base_hash and recorded and recorded != base_hash:
            other_base += 1
    if broken:
        return FAIL, f"저장된 헤드 {len(files)}개 중 {len(broken)}개가 이 컴퓨터의 인코더와 짝이 맞지 않습니다 ({broken[0]})", (
            "헤드는 학습에 쓴 인코더(GGUF, 풀링, 길이)와 같은 것으로만 쓸 수 있습니다. 학습 때와 같은 Qwen3-8B-Q4_K_M.gguf를 쓰거나, 인코더를 바꿨다면 헤드를 다시 학습하세요.")
    note = "" if gguf else " (GGUF가 없어 파일 크기는 비교하지 못했습니다)"
    if other_base:
        return WARN, f"{len(files)}개가 인코더와 짝이 맞지만, {other_base}개는 지금 있는 공식 헤드와 다른 헤드에서 시작해 학습한 것입니다{note}", "구동에는 문제없습니다. 같은 결과를 다시 만들려면 학습 때의 공식 헤드가 필요합니다."
    return OK, f"{len(files)}개 모두 이 컴퓨터의 인코더와 짝이 맞습니다 (GGUF 파일 이름과 크기, 풀링, 길이, 토크나이저){note}", ""


def check_network():
    import requests
    try:
        requests.head("https://huggingface.co", timeout=8)
    except requests.RequestException as error:
        return FAIL, f"{', '.join(ctx['downloads'])}을(를) 내려받아야 하는데 huggingface.co에 연결할 수 없습니다 ({type(error).__name__})", \
            "인터넷 연결을 확인하거나, 다른 컴퓨터의 HF 캐시(~/.cache/huggingface)를 복사해 오세요."
    return OK, "huggingface.co에 연결됩니다 (내려받을 파일: " + ", ".join(ctx["downloads"]) + ")", ""


# ═════════════════════════ 5. 파인튜닝 준비 ═════════════════════════
def load_shared():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "baseline" / "steps"))
    import baseline_bert_config as cfg
    import common
    from util import data_convert as dc
    ctx.update(cfg=cfg, common=common, dc=dc)
    return OK, "baseline_bert_config, common, util.data_convert", ""


def check_datasets():
    cfg, common, dc = ctx["cfg"], ctx["common"], ctx["dc"]
    ready, missing = [], []
    for topic, dataset in common.all_pairs():
        if not common.is_enabled(topic, dataset):
            continue
        lacking = common.missing_files(topic, dataset)
        if lacking:
            missing.append(f"{topic}/{dataset}({lacking[0].name})")
        else:
            ready.append((topic, dataset))
    if not ready:
        return FAIL, "사용할 수 있는 데이터셋이 없습니다", "dataset/<topic>/<데이터셋>/ 의 train/validation/test.csv와 labels.json, baseline_bert_config.py의 DATASET_SWITCH를 확인하세요."
    for topic in cfg.TOPICS:
        first = next((d for t, d in ready if t == topic), None)
        if first is None:
            continue
        labels_path = common.dataset_paths(topic, first)["labels"]
        try:
            dc.load_questions(labels_path, topic)
        except KeyError:
            return FAIL, f"{labels_path} 에 descriptions(선택지 설명)가 없습니다", "CLM은 라벨마다 설명 문장이 필요합니다."
        if common.read_labels(labels_path) != cfg.LABELS[topic]:
            return FAIL, f"{labels_path} 의 라벨 순서가 baseline_bert_config.LABELS와 다릅니다", ""
    ctx["pairs"] = ready
    if missing:
        return WARN, f"{len(ready)}개 사용 가능, 파일이 빠진 데이터셋 {len(missing)}개: {', '.join(missing[:4])}", "빠진 파일을 채우거나 DATASET_SWITCH에서 끄세요."
    return OK, f"{len(ready)}개 데이터셋 (topic {len(cfg.TOPICS)}개) | 질문과 선택지 설명, 라벨 순서 확인", ""


def check_text_lengths():
    S, cfg, common, dc = ctx["S"], ctx["cfg"], ctx["common"], ctx["dc"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S.ENCODER["tokenizer"], local_files_only=True)
    unique = {}
    for topic, dataset in ctx["pairs"]:
        paths = common.dataset_paths(topic, dataset)
        for split in ("train", "validation", "test"):          # 길이만 센다. 평가에는 쓰지 않는다.
            for row in common.read_split(paths[split]):
                unique[(topic, row[cfg.TEXT_COLUMN])] = None
    question = {topic: len(tok("\n\n" + dc.INSTRUCTIONS.get(topic, dc.DEFAULT_INSTRUCTION), add_special_tokens=False)["input_ids"])
                for topic in cfg.TOPICS}
    keys = list(unique)
    counts = [len(ids) + question[topic] for (topic, _), ids in
              zip(keys, tok([text for _, text in keys], add_special_tokens=False)["input_ids"])]
    longest, total = max(counts), sum(counts)
    ctx["token_stats"] = {"longest": longest, "total": total, "texts": len(counts)}
    cap, ubatch = S.ENCODER["ctx"] - 1, S.ENCODER["ubatch"]
    too_long = sum(1 for n in counts if n > cap)
    if too_long:
        return WARN, f"{too_long}개 글이 한도({cap}토큰)보다 길어서 앞부분이 잘립니다 (질문은 끝에 있어서 남습니다)", ""
    detail = f"서로 다른 문장 {len(counts):,}개, 가장 긴 글 {longest}토큰 (한도 {cap}), 전체 약 {total:,}토큰"
    longer_than_ub = sum(1 for n in counts if n > ubatch)
    if longer_than_ub:       # -ub보다 긴 글은 조각으로 나눠 처리된다. 실패가 아니다. (clm_settings.py의 ubatch 주석 참고)
        detail += f" | -ub({ubatch})보다 긴 글 {longer_than_ub}개는 조각으로 나눠 처리됩니다 (결과는 거의 같음)"
    return OK, detail, ""


def check_writable():
    S = ctx["S"]
    problems = []
    for path in (S.EMBED_CACHE_DIR, S.SERVER_LOG_DIR, ROOT / "artifacts", ROOT / "logs"):
        target = nearest_existing(path)
        try:
            with tempfile.NamedTemporaryFile(dir=target):
                pass
        except OSError as error:
            problems.append(f"{target} ({type(error).__name__})")
    if problems:
        return FAIL, "쓸 수 없는 폴더: " + ", ".join(problems), "폴더 권한이나 읽기 전용 설정을 확인하세요."
    return OK, "임베딩 캐시(jev/tmp), 서버 로그, artifacts, logs 폴더에 쓸 수 있습니다", ""


def check_disk():
    from huggingface_hub import constants
    problems, notes = [], []
    project_free = shutil.disk_usage(nearest_existing(ROOT)).free / 2**30
    notes.append(f"프로젝트 드라이브 여유 {project_free:.0f}GB")
    if project_free < MIN_FREE_DISK_GB:
        problems.append(f"프로젝트 드라이브 여유가 {MIN_FREE_DISK_GB}GB보다 적습니다")
    cache = nearest_existing(Path(constants.HF_HUB_CACHE))
    cache_free = shutil.disk_usage(cache).free / 2**30
    need = sum(DOWNLOAD_GB[name] for name in ctx["downloads"]) + (1 if ctx["downloads"] else 0)
    notes.append(f"HF 캐시 드라이브 여유 {cache_free:.0f}GB" + (f" (내려받을 파일에 약 {need:.1f}GB 필요)" if ctx["downloads"] else ""))
    if ctx["downloads"] and cache_free < need:
        return FAIL, " | ".join(notes), "내려받을 공간이 부족합니다. 공간을 비우거나 HF_HOME 환경 변수로 다른 드라이브를 지정하세요."
    if problems:
        return WARN, " | ".join(notes), "결과(artifacts)와 임베딩 캐시를 둘 공간을 확보하세요."
    return OK, " | ".join(notes), ""


def check_training_smoke():
    """공식 헤드를 불러와 합성 데이터로 InfoNCE 학습 몇 스텝과 저장/불러오기를 해 본다. (CPU, 몇 초)"""
    H = ctx["H"]
    import torch
    import torch.nn.functional as F
    started = time.perf_counter()
    model = H.HeadModel.from_checkpoint(ctx["head"], "cpu")
    generator = torch.Generator().manual_seed(0)
    n, k = 64, 3
    labels = torch.randint(0, k, (n,), generator=generator)
    emb = F.normalize(torch.randn(n + k, 4096, generator=generator), dim=-1)
    state_idx, option_idx = torch.arange(n), (n + torch.arange(k)).repeat(n, 1)
    option_target = F.one_hot(labels, k).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    losses = []
    for _ in range(3):
        model.train()
        loss = H.infonce_loss(model, emb, state_idx, option_idx, option_target, torch.arange(n))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    if not all(math.isfinite(value) for value in losses):
        return FAIL, f"학습 손실이 유한하지 않습니다 {losses}", ""
    with tempfile.TemporaryDirectory() as folder:
        model.save(Path(folder))
        loaded = H.HeadModel.load(Path(folder), "cpu")
        before, _ = model.score(emb[:n].numpy(), emb[n:].numpy())
        after, _ = loaded.score(emb[:n].numpy(), emb[n:].numpy())
    if not torch.equal(before, after):
        return FAIL, "저장했다가 불러온 헤드의 점수가 다릅니다", ""
    return OK, (f"헤드 {model.num_parameters():,}개 파라미터: 학습 3스텝(손실 {losses[0]:.2f} -> {losses[-1]:.2f}), "
                f"safetensors 저장/불러오기 일치 ({time.perf_counter() - started:.1f}초)"), ""


# ═════════════════════════ 6. 인코더 서버 ═════════════════════════
def estimate_vram_mb(ubatch: int) -> int:
    """서버(-ngl 99 -c 2048 -np 1)가 쓰는 VRAM 추정(MB). RTX 5060 Laptop에서 -ub별로 잰 값(2048: 5301, 1024: 5085, 512: 4973, 256: 4917)에 맞춘 직선이다."""
    return round(4850 + 0.22 * ubatch)


def judge_vram(free_mb: int, gpu_layers: int, ubatch: int) -> tuple[str, str, str]:
    need_mb = estimate_vram_mb(ubatch)
    if gpu_layers < GPU_LAYERS_FULL:
        return INFO, f"GPU에 레이어 {gpu_layers}/{GPU_LAYERS_FULL}개만 올리는 설정이라 필요량이 더 적습니다 (여유 {free_mb}MiB)", ""
    if free_mb >= need_mb * 1.1:
        return OK, f"여유 {free_mb}MiB (서버가 약 {need_mb}MiB 사용)", ""
    if free_mb >= need_mb:
        return WARN, f"여유 {free_mb}MiB로 빠듯합니다 (서버가 약 {need_mb}MiB 사용)", \
            "다른 GPU 프로그램을 닫거나 .env의 CLM_UBATCH를 512로 줄이세요 (약 0.3GB 절약)."
    return FAIL, f"VRAM 여유가 부족합니다 ({free_mb}MiB < 서버가 쓰는 약 {need_mb}MiB)", (
        "브라우저, 게임 등 GPU를 쓰는 프로그램을 닫으세요. 그래도 부족하면 .env의 CLM_GPU_LAYERS를 줄이세요(예: 30, 대신 느려집니다).\n"
        "CLM_UBATCH를 512로 줄이면 약 0.3GB 절약됩니다.")


def check_server_state():
    S, E = ctx["S"], ctx["E"]
    base = S.ENCODER["url"].rstrip("/")
    parsed = urlparse(base)
    state = E.server_state(base)
    ctx["server_state"] = state
    if state == "foreign":
        return FAIL, f"{base} 를 llama-server가 아닌 다른 프로그램이 쓰고 있습니다", ".env의 CLM_ENCODER_URL로 다른 포트를 지정하세요. (예: http://127.0.0.1:8091)"
    if state in ("ready", "loading"):
        model_path = E.server_identity(base)["model_path"]
        name = Path(model_path).name if model_path else None
        if name and name != S.ENCODER["gguf_file"]:
            return FAIL, f"{base} 에서 다른 모델이 실행 중입니다: {name}", f"필요한 모델은 {S.ENCODER['gguf_file']}입니다. 그 서버를 끄거나 .env의 CLM_ENCODER_URL로 다른 포트를 지정하세요."
        ctx["server_ok"] = True
        status = OK if state == "ready" else WARN
        return status, f"{base} 에 서버가 이미 떠 있습니다 ({state}, {name or '모델 이름 확인 불가'}). 점검과 실행에서 이 서버를 그대로 씁니다", \
            ("" if state == "ready" else "다른 곳에서 모델을 올리는 중입니다. 끝난 뒤 다시 점검하세요.")
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        return FAIL, f"{base} 에 연결할 수 없고, 원격 서버는 직접 띄울 수 없습니다", "서버가 켜져 있는지 확인하세요."
    port = parsed.port or 80
    family = socket.AF_INET6 if ":" in parsed.hostname else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((parsed.hostname, port))
        except OSError as error:
            return FAIL, f"포트 {port}를 쓸 수 없습니다 ({error.strerror or error})", "Windows가 예약한 포트이거나 다른 프로그램이 쓰는 중일 수 있습니다. .env의 CLM_ENCODER_URL로 다른 포트를 지정하세요."
    ctx["server_ok"] = True
    return OK, f"{base} 는 비어 있고 포트를 쓸 수 있습니다 (서버가 필요하면 코드가 직접 띄웁니다)", ""


def check_vram():
    S = ctx["S"]
    if ctx.get("server_state") in ("ready", "loading"):
        return INFO, "서버가 이미 떠 있어서 VRAM 여유는 따로 따지지 않습니다", ""
    return judge_vram(ctx["devices"][0][3], S.ENCODER["gpu_layers"], S.ENCODER["ubatch"])


def embed_ids(enc, id_lists):
    with contextlib.redirect_stdout(io.StringIO()):          # 공식 클라이언트의 진행 출력을 가린다
        return enc.backend.embed(id_lists)


def run_live_check():
    """서버를 (없으면 임시로) 띄워 기준 벡터와 비교하고, 가장 긴 글과 속도를 확인한다. 캐시에는 아무것도 넣지 않는다."""
    names = {"server": "llama-server", "devices": "GPU 인식", "gguf": "인코더 GGUF", "tokenizer": "토크나이저",
             "clm": "CLM 저장소", "server_ok": "포트와 서버 상태"}
    lacking = [label for key, label in names.items() if key not in ctx]
    if lacking:
        report("run", SKIP, "인코더 서버 실제 구동", f"통과하지 못한 점검이 있어 건너뜁니다: {', '.join(lacking)}")
        return
    E = ctx["E"]
    ctx["live_attempted"] = True
    began = time.perf_counter()
    try:
        with E.ensure_encoder() as enc:
            ready = time.perf_counter() - began
            report("run", OK, "인코더 서버", (f"직접 띄웠습니다 ({ready:.0f}초 만에 준비)" if enc.started else "이미 떠 있는 서버를 썼습니다")
                   + f" | llama.cpp {enc.identity.get('build') or '버전 확인 불가'}, 모델 {enc.identity.get('n_params') or '?'}개 파라미터")
            if enc.reference_cosine is None:
                report("run", WARN, "기준 벡터 비교", "기준 벡터가 없거나 다른 설정용이라 비교하지 못했습니다",
                       "서버를 직접 띄운 게 아니라면 --embeddings --pooling last 로 띄웠는지 확인하세요.")
            else:
                report("run", OK, "기준 벡터 비교", f"코사인 {enc.reference_cosine:.5f} (같은 모델과 풀링이면 0.999 이상, 0.99 미만이면 거부)")
            # 가장 긴 글: 데이터의 최대 길이만큼의 토큰을 한 번에 처리할 수 있는지
            stats = ctx.get("token_stats")
            length = min(stats["longest"] if stats else DEFAULT_LONG_TOKENS, enc.token_cap)
            long_ids = enc.recipe.text_ids("이 문서는 점검용으로 만든 문장입니다. " * 400, keep="head")[:length]
            embed_ids(enc, [long_ids[:64]])             # 처음 한 번은 GPU 커널 준비 때문에 느릴 수 있어 시간에서 뺀다
            seconds = []
            for _ in range(3):
                moment = time.perf_counter()
                vector = embed_ids(enc, [long_ids])
                seconds.append(time.perf_counter() - moment)
            if vector.shape != (1, 4096):
                report("run", FAIL, "긴 글 처리", f"임베딩 모양이 {vector.shape}입니다", "")
            else:
                speed = len(long_ids) / sorted(seconds)[1]
                report("run", OK, "긴 글 처리", f"{len(long_ids)}토큰 한 문장을 {sorted(seconds)[1]:.2f}초에 처리 (약 {speed:,.0f}토큰/초)")
                if stats:
                    minutes = stats["total"] / speed / 60
                    report("train", INFO, "예상 임베딩 시간", f"전체 {stats['texts']:,}문장({stats['total']:,}토큰)을 처음 한 번 계산하는 데 약 {minutes:.0f}분 "
                           "(한 번 계산한 문장은 캐시에 남아 다음부터는 건너뜁니다)")
            used, scope = enc.gpu_memory()
            if used is not None:
                report("run", INFO, "서버의 VRAM 사용량", f"약 {used:,.0f}MB ({'서버를 띄우기 전과의 차이' if scope == 'server_delta' else 'GPU 전체 사용량'})")
    except Exception as error:
        lines = str(error).splitlines() or [""]
        report("run", FAIL, "인코더 서버", f"{type(error).__name__}: {lines[0]}", "\n".join(lines[1:]))


# ═════════════════════════ 요약 ═════════════════════════
def summarize(target: str, static: bool) -> int:
    def failed(scopes):
        return [name for scope, status, name in results if status == FAIL and scope in scopes]
    warned = [name for _, status, name in results if status == WARN]
    run_failed, train_failed = failed(("run",)), failed(("run", "train"))
    downloads = ctx["downloads"]

    def verdict(failures, extra=""):
        if failures:
            return f"불가 (실패 {len(failures)}건: {', '.join(dict.fromkeys(failures))})"
        return "가능" + extra

    live = ctx.get("live_attempted", False)
    unverified = "" if live else " (인코더 서버 실행은 확인하지 않음)"
    flush_skips()
    print("\n[요약]")
    print(f"  구동(서빙, 평가)  : {verdict(run_failed, unverified)}")
    print(f"  파인튜닝          : {verdict(train_failed, unverified)}")
    if warned:
        print(f"  경고 {len(warned)}건        : {', '.join(dict.fromkeys(warned))}")
    if downloads:
        print(f"  처음 실행하면 자동으로 내려받는 파일: {', '.join(downloads)}")
    if static:
        print("  --static이라 서버를 띄우지 않았습니다. 인코더가 실제로 도는지는 옵션 없이 다시 실행해서 확인하세요.")
    elif not live:
        print("  앞 단계가 준비되지 않아 인코더 서버가 실제로 도는지는 확인하지 못했습니다. 위의 경고와 실패를 해결한 뒤 다시 실행하세요.")
    return 1 if (run_failed if target == "run" else train_failed) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CLM을 구동하거나 파인튜닝할 수 있는 컴퓨터인지 점검")
    parser.add_argument("--static", action="store_true", help="인코더 서버를 띄우지 않고 설치와 파일 상태만 점검")
    parser.add_argument("--for", dest="target", choices=("run", "train"), default="train",
                        help="종료 코드를 정하는 기준. run: 구동만, train: 구동과 파인튜닝 (기본)")
    args = parser.parse_args()
    print(f"CLM 환경 점검 | {ROOT} | {'정적 점검(--static)' if args.static else '전체 점검'}")

    section("1. 컴퓨터")
    run_check("run", "운영체제, 메모리, GPU", check_machine)

    section("2. 파이썬 환경")
    run_check("run", "파이썬", check_python)
    run_check("run", "필수 패키지", check_packages)
    run_check("run", "프로젝트 모듈", load_project, needs=("packages",))
    run_check("info", "torch", check_torch, needs=("modules",))

    section("3. 외부 도구")
    run_check("run", "CLM 저장소", check_clm_repo, needs=("modules",))
    run_check("run", "llama-server", check_llama_server, needs=("modules",))
    run_check("run", "GPU 인식", check_gpu_devices, needs=("server",))

    section("4. 모델 파일 (점검은 내려받지 않습니다)")
    run_check("run", "인코더 GGUF", check_gguf, needs=("modules",))
    run_check("run", "공식 헤드", check_base_head, needs=("modules",))
    run_check("run", "토크나이저", check_tokenizer, needs=("modules",))
    run_check("run", "저장된 헤드와 인코더의 짝", check_saved_heads, needs=("modules",))
    if ctx["downloads"]:
        run_check("run", "내려받기 연결", check_network, needs=("modules",))

    section("5. 파인튜닝 준비")
    run_check("train", "공통 코드와 설정", load_shared, needs=("modules",))
    run_check("train", "데이터셋", check_datasets, needs=("common",))
    run_check("train", "입력 길이", check_text_lengths, needs=("pairs", "tokenizer"))
    run_check("train", "쓰기 권한", check_writable, needs=("modules",))
    run_check("train", "저장 공간", check_disk, needs=("modules",))
    run_check("train", "헤드 학습 동작", check_training_smoke, needs=("head", "modules", "clm"))

    section("6. 인코더 서버")
    run_check("run", "포트와 서버 상태", check_server_state, needs=("modules",))
    run_check("run", "VRAM 여유", check_vram, needs=("devices", "server_state"))
    flush_skips()
    if args.static:
        report("run", SKIP, "인코더 서버 실제 구동", "--static이라 건너뜁니다")
    else:
        run_live_check()
    return summarize(args.target, args.static)


if __name__ == "__main__":
    sys.exit(main())
