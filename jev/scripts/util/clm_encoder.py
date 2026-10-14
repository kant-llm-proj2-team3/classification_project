"""
CLM 인코더 관리와 문장 임베딩 계산

인코더 = Qwen3-8B Q4_K_M GGUF를 llama-server(--embeddings --pooling last)로 띄운 임베딩 서버.
  ensure_encoder() : 서버가 이미 떠 있으면 같은 모델인지 확인해서 그대로 쓰고, 없으면 직접 띄운다.
                     직접 띄운 서버는 with 블록이 끝나면 끈다. (다른 사람이 띄운 서버는 건드리지 않는다)
  Encoder.embed()  : 문장 -> 4096차원 벡터(L2 정규화). 계산한 벡터는 디스크 캐시에 쌓아 같은 문장은 다시 계산하지 않는다.
토큰화 규칙(Recipe)과 서버 클라이언트(ServerBackend)는 CLM 공식 코드를 그대로 쓴다.
프로젝트 밖의 CLM 저장소를 sys.path로 불러오며, 경로와 설정은 clm_settings.py에 있다.

사용 예
    with ensure_encoder() as enc:
        vectors, seconds = enc.embed(["문장 1", "문장 2"])
"""

import atexit
import base64
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import numpy as np
import requests

import clm_settings as S

HIDDEN = 4096          # Qwen3-8B 임베딩 차원 (공식 헤드의 입력 차원)
# 기준 벡터: 올바르게 설정한 서버가 PROBE_TEXT에 대해 돌려준 임베딩. 서버를 연결할 때마다 같은 결과가 나오는지 비교한다.
# 같은 GGUF는 GPU/CPU/재시작과 상관없이 코사인 0.9993 이상이고, 풀링이 mean이면 0.78~0.93이라서 0.99로 구분한다.
REFERENCE_FILE = Path(__file__).with_name("clm_reference.json")
PROBE_TEXT = "연결 확인용 문장입니다."
_CLM = None


# ═════════════════════════ 1. CLM 공식 코드 불러오기 ═════════════════════════
def _git_commit(repo: Path) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def load_clm() -> SimpleNamespace:
    """CLM 저장소의 헤드 구조(heads), 질문->텍스트 규칙(schema), 토큰화와 서버 클라이언트(embed_utils)."""
    global _CLM
    if _CLM is not None:
        return _CLM
    repo = S.CLM_REPO
    if not (repo / "src" / "clm" / "heads.py").exists():
        raise FileNotFoundError(
            f"CLM 저장소를 찾을 수 없습니다: {repo}\n"
            f"  git clone https://github.com/Contrastive-LM/CLM.git \"{repo}\"\n"
            f"  git -C \"{repo}\" checkout {S.CLM_REPO_COMMIT}\n"
            "  다른 위치에 두었다면 .env의 CLM_REPO에 경로를 적으세요.")
    commit = _git_commit(repo)
    if commit and commit != S.CLM_REPO_COMMIT:
        print(f"  [주의] CLM 저장소 커밋({commit[:7]})이 검증한 커밋({S.CLM_REPO_COMMIT[:7]})과 다릅니다. "
              "학습 방식이 바뀌었을 수 있습니다.")
    for sub in ("src", "train"):
        path = str(repo / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    import embed_utils
    from clm import heads, schema
    _CLM = SimpleNamespace(embed_utils=embed_utils, heads=heads, schema=schema, commit=commit)
    return _CLM


# ═════════════════════════ 2. 임베딩 캐시 ═════════════════════════
class VectorCache:
    """문장(sha1) -> (임베딩 float16, 인코더 계산 시간). 끝에 덧붙이기만 하는 파일이라 중간에 끊겨도 앞부분은 남는다.

    keys.txt(한 줄에 sha1 하나) / vectors.f16(float16 행렬, 4096열) / seconds.f32(문장별 인코더 계산 시간)
    세 파일의 행 수가 다르면(저장 도중 끊긴 경우) 가장 짧은 쪽에 맞춰 잘라 낸다.
    """

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._keys_path = self.dir / "keys.txt"
        self._vecs_path = self.dir / "vectors.f16"
        self._secs_path = self.dir / "seconds.f32"

        keys = self._read_keys()
        vecs = np.fromfile(self._vecs_path, dtype="<f2") if self._vecs_path.exists() else np.empty(0, "<f2")
        secs = np.fromfile(self._secs_path, dtype="<f4") if self._secs_path.exists() else np.empty(0, "<f4")
        n = min(len(keys), len(vecs) // HIDDEN, len(secs))
        self._trim(len(keys), n)
        vecs = vecs[: n * HIDDEN].reshape(n, HIDDEN)
        self._rows = {key: (vecs[i], float(secs[i])) for i, key in enumerate(keys[:n])}

    def _read_keys(self) -> list[str]:
        if not self._keys_path.exists():
            return []
        text = self._keys_path.read_text(encoding="ascii", errors="ignore")
        lines = text.split("\n")
        if not text.endswith("\n"):
            lines = lines[:-1]                 # 마지막 줄이 쓰다 만 것이면 버린다
        keys = []
        for line in lines:
            if len(line) != 40:                # sha1이 아니면 거기서 멈춘다
                break
            keys.append(line)
        return keys

    def _trim(self, key_count: int, n: int) -> None:
        for path, size in ((self._vecs_path, n * HIDDEN * 2), (self._secs_path, n * 4)):
            if path.exists() and path.stat().st_size != size:
                with path.open("r+b") as stream:
                    stream.truncate(size)
        if key_count != n or (self._keys_path.exists() and self._keys_path.stat().st_size != n * 41):
            keys = self._read_keys()[:n]
            self._keys_path.write_text("".join(key + "\n" for key in keys), encoding="ascii", newline="\n")

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def __contains__(self, key: str) -> bool:
        return key in self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def vector(self, key: str) -> np.ndarray:
        return self._rows[key][0]

    def seconds(self, key: str) -> float:
        return self._rows[key][1]

    def append(self, keys: list[str], vectors: np.ndarray, seconds: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype="<f2")
        seconds = np.asarray(seconds, dtype="<f4")
        with self._vecs_path.open("ab") as stream:     # 벡터와 시간을 먼저 쓰고 키를 마지막에 쓴다. (끊겨도 키가 있는 행은 항상 온전함)
            stream.write(vectors.tobytes())
        with self._secs_path.open("ab") as stream:
            stream.write(seconds.tobytes())
        with self._keys_path.open("a", encoding="ascii", newline="\n") as stream:
            stream.write("".join(key + "\n" for key in keys))
        for key, vector, second in zip(keys, vectors, seconds, strict=True):
            self._rows[key] = (vector, float(second))


# ═════════════════════════ 3. llama-server 찾기 · 띄우기 · 확인 ═════════════════════════
def find_llama_server() -> Path | None:
    if S.LLAMA_SERVER:
        path = Path(S.LLAMA_SERVER)
        if not path.is_file():
            raise FileNotFoundError(f".env의 CLM_LLAMA_SERVER 경로에 파일이 없습니다: {path}")
        return path
    found = shutil.which("llama-server")
    candidates = ([Path(found)] if found else []) + [S.TOOLS_DIR / "llama.cpp" / name
                                                     for name in ("llama-server.exe", "llama-server")]
    return next((path for path in candidates if path.is_file()), None)


def resolve_gguf(download: bool) -> Path | None:
    """GGUF 파일 경로(HF 캐시). download=False면 이미 받은 것만 찾고, 없으면 None."""
    from huggingface_hub import hf_hub_download
    enc = S.ENCODER
    try:
        return Path(hf_hub_download(enc["gguf_repo"], enc["gguf_file"], local_files_only=True))
    except Exception:
        if not download:
            return None
    print(f"  GGUF를 내려받습니다: {enc['gguf_repo']}/{enc['gguf_file']} (약 5GB, HF 캐시에 저장)")
    return Path(hf_hub_download(enc["gguf_repo"], enc["gguf_file"]))


def _get(url: str, timeout: float = 3.0):
    try:
        return requests.get(url, timeout=timeout)
    except requests.RequestException:
        return None


def server_state(base: str) -> str | None:
    """None: 연결 안 됨 / 'ready' / 'loading' / 'foreign': 그 주소를 llama-server가 아닌 것이 쓰고 있음."""
    response = _get(base + "/health")
    if response is None:
        return None
    return {200: "ready", 503: "loading"}.get(response.status_code, "foreign")


def server_identity(base: str) -> dict:
    """떠 있는 서버가 알려 주는 모델 정보. 알 수 없는 값은 None."""
    info = {"model_path": None, "build": None, "n_params": None, "size_bytes": None}
    response = _get(base + "/props", 10)
    if response is not None and response.status_code == 200:
        body = response.json()
        info["model_path"], info["build"] = body.get("model_path"), body.get("build_info")
    response = _get(base + "/v1/models", 10)
    if response is not None and response.status_code == 200:
        data = (response.json().get("data") or [{}])[0]
        meta = data.get("meta") or {}
        info["n_params"], info["size_bytes"] = meta.get("n_params"), meta.get("size")
    return info


def gpu_used_mb() -> float | None:
    """GPU 0번이 지금 쓰는 메모리(MB). 모든 프로그램의 합이다. nvidia-smi가 없으면 None."""
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                              capture_output=True, text=True, timeout=10)
        return float(done.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _tail(path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(로그를 읽지 못했습니다)"


def _start_server(base: str) -> tuple[subprocess.Popen, Path]:
    enc = S.ENCODER
    parsed = urlparse(base)
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError(f"인코더 서버({base})에 연결할 수 없습니다. 원격 서버는 직접 띄울 수 없으니 서버 상태를 확인하세요.")
    server = find_llama_server()
    if server is None:
        raise FileNotFoundError(
            "llama-server를 찾을 수 없습니다. llama.cpp의 Windows CUDA 빌드(llama-bXXXXX-bin-win-cuda-12.4-x64.zip과 "
            f"cudart-llama-bin-win-cuda-12.4-x64.zip)를 같은 폴더에 풀어 {S.TOOLS_DIR / 'llama.cpp'} 에 두거나,\n"
            "  .env의 CLM_LLAMA_SERVER에 llama-server 경로를 적으세요. (https://github.com/ggml-org/llama.cpp/releases)")
    gguf = resolve_gguf(download=True)
    command = [str(server), "-m", str(gguf), "--embeddings", "--pooling", enc["pooling"],
               "-c", str(enc["ctx"]), "-b", str(enc["batch"]), "-ub", str(enc["ubatch"]), "-np", str(enc["parallel"]),
               "-ngl", str(enc["gpu_layers"]), "--alias", enc["served_name"],
               "--host", "127.0.0.1", "--port", str(parsed.port or 8090)]
    S.SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = S.SERVER_LOG_DIR / f"llama-server_{time.strftime('%Y%m%d_%H%M%S')}.log"
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if sys.platform == "win32" else 0
    print(f"  인코더 서버를 띄웁니다: {server.name} | {gguf.name} | ngl {enc['gpu_layers']} ub {enc['ubatch']}")
    print(f"  서버 로그: {log_path}")
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=server.parent, creationflags=flags)
    return process, log_path


def _wait_ready(base: str, process: subprocess.Popen | None, log_path: Path | None, timeout: float) -> None:
    started = time.time()
    while time.time() - started < timeout:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"llama-server가 시작 도중 종료됐습니다 (종료 코드 {process.returncode}). 로그 마지막 부분:\n"
                f"{_tail(log_path)}\n"
                "  VRAM이 부족하면 .env의 CLM_GPU_LAYERS(예: 30)나 CLM_UBATCH(예: 1536)를 줄여 보세요.")
        if server_state(base) == "ready":
            return
        time.sleep(1)
    raise TimeoutError(f"인코더 서버가 {timeout:.0f}초 안에 준비되지 않았습니다." +
                       (f" 로그: {log_path}" if log_path else ""))


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


# ═════════════════════════ 4. 인코더 ═════════════════════════
class Encoder:
    """문장 -> 임베딩. 공식 토큰화 규칙(Recipe)으로 토큰 ID를 만들어 서버에 보내고, 결과를 디스크 캐시에 쌓는다."""

    def __init__(self, base: str, identity: dict, started: bool, gpu_baseline_mb: float | None):
        enc = S.ENCODER
        clm = load_clm()
        self.base, self.started = base, started
        self.recipe = clm.embed_utils.Recipe(enc["tokenizer"], enc["ctx"])
        self.backend = clm.embed_utils.ServerBackend(base + "/v1/embeddings", enc["served_name"],
                                                     batch=enc["request_chunk"])
        self.token_cap = enc["ctx"] - 1                       # Recipe가 남기는 최대 토큰 수
        self.identity = identity
        self.clm_commit = clm.commit
        self._gpu_baseline = gpu_baseline_mb
        self.gpu_peak_mb = None
        self.reference_cosine = None          # 기준 벡터와 비교한 결과. 비교하지 않았으면 None
        self.probe_vector = None

        gguf = resolve_gguf(download=False)
        self.gguf_size = gguf.stat().st_size if gguf else identity.get("size_bytes")
        self.cache = VectorCache(S.EMBED_CACHE_DIR / f"{Path(enc['gguf_file']).stem}_ctx{enc['ctx']}")
        self._check_cache_identity()

    def _check_cache_identity(self) -> None:
        meta_path = self.cache.dir / "meta.json"
        meta = {"gguf_file": S.ENCODER["gguf_file"], "gguf_size_bytes": self.gguf_size, "ctx": S.ENCODER["ctx"],
                "pooling": S.ENCODER["pooling"], "tokenizer": S.ENCODER["tokenizer"], "hidden_size": HIDDEN}
        old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        old_size, new_size = old.get("gguf_size_bytes"), meta["gguf_size_bytes"]
        if old_size and new_size and old_size != new_size:
            raise RuntimeError(f"임베딩 캐시({self.cache.dir})가 만들어질 때와 GGUF 파일 크기가 다릅니다. "
                               "같은 이름의 다른 파일이면 캐시 폴더를 지우고 다시 만드세요.")
        if not old or (not old_size and new_size):       # 처음이거나, 크기를 이제야 알게 된 경우에만 기록한다
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def check_connection(self, verify: bool = True) -> None:
        """짧은 문장 하나로 서버가 4096차원 임베딩을 돌려주는지, 기준 벡터와 같은 결과인지 확인한다. (캐시에는 넣지 않는다)"""
        ids = self.recipe.text_ids(PROBE_TEXT, keep="tail")
        with contextlib.redirect_stdout(io.StringIO()):
            vector = self.backend.embed([ids])
        if vector.shape != (1, HIDDEN):
            raise RuntimeError(f"임베딩 차원이 {vector.shape[-1]}입니다. {HIDDEN}차원이어야 합니다. 서버의 모델을 확인하세요. "
                               f"(서버 옵션: --embeddings --pooling {S.ENCODER['pooling']})")
        self.probe_vector = vector[0]
        if verify:
            self._check_reference(vector[0])

    def _check_reference(self, vector: np.ndarray) -> None:
        enc = S.ENCODER
        if not REFERENCE_FILE.exists():
            print(f"  [주의] 기준 벡터({REFERENCE_FILE.name})가 없어 인코더 검증을 건너뜁니다.")
            return
        reference = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
        wanted = (reference["gguf_file"], reference["pooling"], reference["max_len"], reference["text"])
        if wanted != (enc["gguf_file"], enc["pooling"], enc["ctx"], PROBE_TEXT):
            print(f"  [주의] 기준 벡터는 다른 설정({reference['gguf_file']}, {reference['pooling']} 풀링, "
                  f"max_len {reference['max_len']})용이라 인코더 검증을 건너뜁니다.")
            return
        expected = np.frombuffer(base64.b64decode(reference["vector_float16_base64"]), dtype="<f2").astype(np.float32)
        cosine = float(vector @ expected / (np.linalg.norm(vector) * np.linalg.norm(expected)))
        self.reference_cosine = cosine
        if cosine < reference["min_cosine"]:
            raise RuntimeError(
                f"인코더가 기준과 다른 임베딩을 돌려줍니다 (기준 벡터와 코사인 {cosine:.4f}, 최소 {reference['min_cosine']}). "
                f"{enc['gguf_file']} 파일과 서버 옵션(--embeddings --pooling {enc['pooling']})을 확인하세요.\n"
                "  다른 양자화나 모델을 일부러 쓰는 것이라면 기준 벡터를 다시 만드세요:\n"
                "  uv run python -c \"import sys; sys.path.insert(0, 'jev/scripts'); "
                "from util.clm_encoder import write_reference; write_reference()\"")

    def describe(self) -> dict:
        """이 인코더를 다시 찾아 쓰기 위한 기록. 헤드와 함께 저장한다. (헤드는 이 인코더와 짝이 맞을 때만 의미가 있다)"""
        enc = S.ENCODER
        return {"encoder": "Qwen3-8B", "quantization": "Q4_K_M", "gguf_repo": enc["gguf_repo"],
                "gguf_file": enc["gguf_file"], "gguf_size_bytes": self.gguf_size, "pooling": enc["pooling"],
                "max_len": enc["ctx"], "hidden_size": HIDDEN, "served_name": enc["served_name"],
                "tokenizer": enc["tokenizer"], "llama_cpp_build": self.identity.get("build"),
                "clm_commit": self.clm_commit, "reference_cosine": self.reference_cosine}

    def token_lengths(self, texts: list[str]) -> list[int]:
        flatten = self.recipe._flatten
        return [len(flatten(self.recipe.tok(text, add_special_tokens=False)["input_ids"])) for text in texts]

    def embed(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """-> ([n, 4096] float32 (L2 정규화), [n] 문장별 인코더 계산 시간(초))
        이미 계산한 문장은 캐시에서 가져오고 그 문장의 기록된 계산 시간을 돌려준다. 시간은 요청 시간을 토큰 수로 나눈 값이다."""
        keys = [self.cache.key(text) for text in texts]
        todo = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in self.cache and key not in todo:
                todo[key] = text
        if todo:
            self._compute(list(todo.items()))
        vectors = np.stack([self.cache.vector(key) for key in keys]).astype(np.float32)
        return vectors, np.array([self.cache.seconds(key) for key in keys])

    def _compute(self, items: list[tuple[str, str]]) -> None:
        chunk = S.ENCODER["request_chunk"]
        total, started, last_print = len(items), time.perf_counter(), 0.0
        print(f"  [embed] 새 문장 {total}개를 인코더로 계산합니다. (캐시 {len(self.cache)}개 재사용 가능)", flush=True)
        for start in range(0, total, chunk):
            part = items[start:start + chunk]
            ids = [self.recipe.text_ids(text, keep="tail") for _, text in part]
            lengths = np.array([len(one) for one in ids], dtype=float)
            moment = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):          # 공식 클라이언트의 진행 출력을 가린다
                vectors = self.backend.embed(ids)
            elapsed = time.perf_counter() - moment
            if vectors.shape[1] != HIDDEN:
                raise RuntimeError(f"임베딩 차원이 {vectors.shape[1]}입니다. {HIDDEN}차원이어야 합니다. 서버 모델을 확인하세요.")
            self.cache.append([key for key, _ in part], vectors, elapsed * lengths / lengths.sum())
            used = gpu_used_mb()
            if used is not None:
                self.gpu_peak_mb = max(self.gpu_peak_mb or 0.0, used)
            now = time.perf_counter() - started
            done = min(start + chunk, total)
            if now - last_print >= 10 or done == total:
                print(f"  [embed] {done}/{total} ({now:.0f}초)", flush=True)
                last_print = now

    def gpu_memory(self) -> tuple[float | None, str | None]:
        """인코더가 쓴 GPU 메모리(MB)와 그 의미. 직접 띄운 서버는 시작 전과의 차이, 그 밖에는 GPU 전체 사용량이다."""
        used = gpu_used_mb()          # 임베딩이 모두 캐시에 있어도 서버가 떠 있는 동안의 사용량은 잴 수 있다
        if used is not None:
            self.gpu_peak_mb = max(self.gpu_peak_mb or 0.0, used)
        if self.gpu_peak_mb is None:
            return None, None
        if self.started and self._gpu_baseline is not None:
            return max(self.gpu_peak_mb - self._gpu_baseline, 0.0), "server_delta"
        return self.gpu_peak_mb, "gpu_total_used"


@contextlib.contextmanager
def ensure_encoder(verify: bool = True):
    """인코더 서버를 준비해서 Encoder를 돌려준다.

    이미 같은 모델이 떠 있으면 그대로 쓰고, 없으면 llama-server를 띄운다. 직접 띄운 서버만 블록이 끝날 때 끈다.
    verify=True면 기준 벡터와 비교해서 다른 모델/양자화/풀링이면 멈춘다. (기준 벡터를 새로 만들 때만 False)
    """
    enc = S.ENCODER
    base = enc["url"].rstrip("/")
    state, process, log_path, baseline = server_state(base), None, None, None
    if state == "foreign":
        raise RuntimeError(f"{base} 를 llama-server가 아닌 다른 프로그램이 쓰고 있습니다. .env의 CLM_ENCODER_URL로 다른 포트를 지정하세요.")
    if state is None:
        baseline = gpu_used_mb()
        process, log_path = _start_server(base)
        atexit.register(_stop_server, process)
    else:
        print(f"  인코더 서버가 이미 떠 있습니다: {base} (상태 {state})")
    try:
        started = time.time()
        _wait_ready(base, process, log_path, enc["start_timeout"])
        identity = server_identity(base)
        model_path = identity["model_path"]
        if model_path and Path(model_path).name != enc["gguf_file"]:
            raise RuntimeError(f"{base} 에서 다른 모델이 실행 중입니다: {Path(model_path).name} (필요한 파일: {enc['gguf_file']}). "
                               "헤드는 학습에 쓴 GGUF와 짝이 맞아야 하므로 그대로 쓸 수 없습니다.")
        if process is not None:
            print(f"  인코더 서버 준비 완료 ({time.time() - started:.0f}초)")
        encoder = Encoder(base, identity, started=process is not None, gpu_baseline_mb=baseline)
        encoder.check_connection(verify)
        yield encoder
    finally:
        if process is not None:
            _stop_server(process)
            atexit.unregister(_stop_server)
            print("  직접 띄운 인코더 서버를 종료했습니다.")


def write_reference() -> None:
    """지금 서버(올바르게 설정한 것)의 임베딩을 기준 벡터로 저장한다. 인코더를 일부러 바꿨을 때만 다시 만든다."""
    with ensure_encoder(verify=False) as enc:
        vector = enc.probe_vector.astype("<f2")
        REFERENCE_FILE.write_text(json.dumps({
            "text": PROBE_TEXT, "gguf_file": S.ENCODER["gguf_file"], "pooling": S.ENCODER["pooling"],
            "max_len": S.ENCODER["ctx"], "llama_cpp_build": enc.identity.get("build"), "min_cosine": 0.99,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "vector_float16_base64": base64.b64encode(vector.tobytes()).decode("ascii")},
            ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  기준 벡터를 저장했습니다: {REFERENCE_FILE}")
