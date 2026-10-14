"""분류 모델 백엔드. best_model 폴더 하나를 불러와 문장 1건씩 분류한다.

  LayaBackend : laya (mmBERT 기반, 인코더 + 판단 헤드 전체가 best_model에 있다)
  ClmBackend  : CLM (판단 헤드만 best_model에 있고, 인코더는 llama.cpp 서버로 띄운 Qwen3-8B Q4)
  HfBackend   : bert / roberta / koelectra (transformers 분류 모델)

모든 백엔드가 같은 predict(text) 결과 형식(PREDICTION_KEYS)을 돌려준다. 어떤 모델을 쓸지는 classifier.py가 정한다.
학습 코드(jev/scripts, baseline/steps)의 로딩·입력 방식을 그대로 써서 학습·평가 때와 같은 값이 나오게 한다.
"""

import contextlib
import io
import json
import sys
import threading
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
JEV_SCRIPTS = ROOT / "jev" / "scripts"
BASELINE_STEPS = ROOT / "baseline" / "steps"

QID = "label"
# 모든 백엔드 predict()가 돌려주는 키: labels(선택지 순서), probabilities(라벨 -> 확률), label, confidence,
# inference_ms(모델 계산 시간), input_tokens(모델이 본 입력 토큰 수), truncated(입력이 잘렸는지), extra(모델별 추가 지표)
PREDICTION_KEYS = ("labels", "probabilities", "label", "confidence", "inference_ms", "input_tokens", "truncated", "extra")


def add_path(path: Path) -> None:
    # 학습 코드의 모듈(clm_settings, util.*, common 등)을 불러오기 위한 경로. 이미 있는 패키지를 가리지 않도록 뒤에 붙인다.
    if str(path) not in sys.path:
        sys.path.append(str(path))


def resolve_device(name: str | None = None) -> torch.device:
    name = name or ("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다.")
    return torch.device(name)


def build_prediction(labels: list[str], probabilities: list[float], seconds: float, input_tokens: int,
                     truncated: bool, extra: dict | None = None) -> dict:
    best = max(range(len(labels)), key=lambda index: probabilities[index])
    return {"labels": labels, "probabilities": dict(zip(labels, probabilities, strict=True)),
            "label": labels[best], "confidence": float(probabilities[best]),
            "inference_ms": seconds * 1000, "input_tokens": input_tokens, "truncated": truncated,
            "extra": extra or {}}


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_labels(saved: list[str], found: list[str], where: str) -> None:
    if saved != found:
        raise ValueError(f"{where}: 학습 때의 라벨 순서 {saved}와 모델의 라벨 순서 {found}가 다릅니다.")


# ═════════════════════════ 1. laya ═════════════════════════
class LayaBackend:
    """laya 체크포인트(best_model)로 분류한다. 학습 평가(step04_train_laya.predict_items)와 같은 계산(온도 보정 없음)이다."""

    def __init__(self, best_dir: Path, device: torch.device):
        from laya.common import state_room
        from laya.train import calibration_records, items_from_rows, load_checkpoint
        self._state_room, self._records, self._items_from_rows = state_room, calibration_records, items_from_rows

        run_config = read_json(best_dir / "run_config.json")
        self.questions = read_json(best_dir / "questions.json")
        self.labels = list(self.questions[QID]["criteria"])
        check_labels(run_config["labels"], self.labels, "laya")

        self.device = device
        self.model, self.tok, cfg = load_checkpoint(str(best_dir))
        self.model.to(device).eval()
        self.max_len = run_config.get("max_length") or cfg.get("max_len", 512)
        self.head_max_len = run_config.get("head_max_len") or cfg.get("head_max_len", 192)
        self._lock = threading.Lock()      # 모델 하나를 동시에 쓰지 않는다. (GPU 추론은 한 번에 하나씩)

    def predict(self, text: str) -> dict:
        # 정답 칸(expected)은 입력 형식을 맞추기 위한 자리이고 점수에는 쓰이지 않는다.
        row = {"state": text, "questions": self.questions, "expected": {QID: self.labels[0]}}
        items, skipped = self._items_from_rows(self.tok, [row], self.max_len, self.head_max_len)
        if skipped or len(items) != 1:
            raise ValueError(f"laya 입력으로 바꾸지 못했습니다: {skipped}")
        item = items[0]
        truncated = len(item["state_ids"]) > self._state_room(self.tok, item["q"], self.max_len, self.head_max_len)

        with self._lock, torch.no_grad():
            started = time.perf_counter()
            records = self._records(self.model, self.tok, items, self.device, self.max_len, self.head_max_len,
                                    batch_size=1)
            seconds = time.perf_counter() - started
        logits = torch.tensor(records[0][1], dtype=torch.float32)
        probabilities = torch.softmax(logits, dim=-1).tolist()
        return build_prediction(self.labels, probabilities, seconds, len(item["state_ids"]), truncated)

    def close(self) -> None:
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


# ═════════════════════════ 2. CLM ═════════════════════════
class ClmBackend:
    """CLM 판단 헤드(best_model) + 외부 인코더 서버로 분류한다.

    인코더 서버(llama.cpp)는 모든 주제가 하나를 같이 쓰므로 classifier.py가 한 번만 띄워서 넘겨 준다.
    서비스에서는 요청 문장이 디스크 임베딩 캐시에 쌓이지 않도록 인코더 서버에 직접 요청하고,
    선택지 설명의 임베딩은 불러올 때 한 번만 계산해 둔다.
    """

    def __init__(self, best_dir: Path, encoder):
        add_path(JEV_SCRIPTS)
        from util.clm_head import ClmClassifier

        self.encoder = encoder
        self.classifier = ClmClassifier.from_best_dir(best_dir, encoder)   # 학습 때의 인코더와 짝이 맞는지도 확인한다.
        self.labels = list(self.classifier.keys)
        check_labels(read_json(best_dir / "run_config.json")["labels"], self.labels, "clm")
        self.candidates = self._embed(self.classifier.candidate_texts)[0]
        self._lock = threading.Lock()

    def _embed(self, texts: list[str]):
        # -> (임베딩 [n, 4096], 인코더 서버 계산 시간(초), 토큰 수 목록)
        ids = [self.encoder.recipe.text_ids(text, keep="tail") for text in texts]
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):          # 공식 클라이언트의 진행 출력을 가린다
            vectors = self.encoder.backend.embed(ids)
        return vectors, time.perf_counter() - started, [len(one) for one in ids]

    def predict(self, text: str) -> dict:
        state_text = self.classifier.state_texts([text])[0]      # 질문(instructions)을 붙인 학습 때와 같은 입력
        truncated = self.encoder.token_lengths([state_text])[0] > self.encoder.token_cap
        with self._lock:
            vectors, encode_seconds, token_counts = self._embed([state_text])
            logits, head_seconds = self.classifier.head.score(vectors, self.candidates)
        probabilities = torch.softmax(logits[0], dim=-1).tolist()
        return build_prediction(self.labels, probabilities, encode_seconds + head_seconds, token_counts[0], truncated,
                                extra={"encode_ms": encode_seconds * 1000, "head_ms": head_seconds * 1000})

    def close(self) -> None:
        self.classifier = None      # 인코더 서버는 classifier.py가 종료한다.


# ═════════════════════════ 3. bert / roberta / koelectra ═════════════════════════
class HfBackend:
    """transformers 분류 모델(best_model). 입력은 baseline/steps/common.encode_text로 만든다. (학습·test 평가와 같은 방식)"""

    def __init__(self, best_dir: Path, device: torch.device):
        add_path(ROOT)
        add_path(BASELINE_STEPS)
        import common
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not (best_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"저장된 가중치가 없습니다: {best_dir / 'model.safetensors'}")
        run_config = read_json(best_dir / "run_config.json")
        self._encode_text = common.encode_text
        self.max_length = run_config["max_length"]
        self.truncation = run_config.get("truncation", "right")
        self.head_tokens = run_config.get("head_tokens")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(best_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            best_dir, local_files_only=True, use_safetensors=True).to(device).eval()
        self.labels = [self.model.config.id2label[index] for index in range(self.model.config.num_labels)]
        check_labels(run_config["labels"], self.labels, "bert 계열")
        self._lock = threading.Lock()

    def predict(self, text: str) -> dict:
        item, truncated = self._encode_text(text, self.tokenizer, self.max_length, self.truncation, self.head_tokens)
        features = {key: torch.tensor([value], device=self.device) for key, value in item.items()}
        with self._lock, torch.inference_mode():
            started = time.perf_counter()
            logits = self.model(**features).logits
            seconds = time.perf_counter() - started
        probabilities = torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()
        return build_prediction(self.labels, probabilities, seconds, len(item["input_ids"]), truncated)

    def close(self) -> None:
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
