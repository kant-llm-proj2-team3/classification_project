"""main.py에서 사용할 분류기 로딩·추론 인터페이스.

서버가 시작될 때 주제(documents, inquiries)마다 모델을 하나씩 불러온다.
어떤 모델을 쓸지는 artifacts/<주제>/advanced_merge/recommendation.json의 recommended_model을 따르고,
그 모델의 best_model 폴더(같은 advanced_merge 안)를 불러온다.
.env의 CLASSIFIER_MODEL_DOCUMENTS / CLASSIFIER_MODEL_INQUIRIES로 모델을 바꿔 쓸 수 있다. (laya, clm, bert, roberta, koelectra)
"""

import contextlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from classifier_backends import JEV_SCRIPTS, ClmBackend, HfBackend, LayaBackend, add_path, resolve_device
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)

ARTIFACTS = ROOT / "artifacts"
TOPICS = ("documents", "inquiries")
# 가장 어려운 데이터라서 이 데이터셋의 비교 결과(recommendation.json)로 서비스 모델을 정한다.
RECOMMEND_DATASET = "advanced_merge"
HF_KEYS = ("bert", "roberta", "koelectra")
SUPPORTED_KEYS = ("laya", "clm", *HF_KEYS)

logger = logging.getLogger("uvicorn.error")
# 모든 주제가 같이 쓰는 외부 자원(CLM 인코더 서버). 서버가 끝날 때 close_classifiers()가 정리한다.
_resources = contextlib.ExitStack()
_encoder = None


@dataclass
class ClassifierBundle:
    topic: str
    model_key: str | None = None
    run_name: str | None = None
    backend: object = None
    error: str | None = None       # 불러오지 못했을 때의 이유. 있으면 이 주제는 분류하지 않는다.

    @property
    def labels(self) -> list[str]:
        return list(getattr(self.backend, "labels", []))


def read_recommendation(topic: str) -> dict:
    path = ARTIFACTS / topic / RECOMMEND_DATASET / "recommendation.json"
    if not path.exists():
        raise FileNotFoundError(f"추천 결과가 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def choose_model(topic: str) -> tuple[str, str | None, Path]:
    """-> (모델 키, 추천 때의 run 이름(직접 지정했으면 None), best_model 폴더)"""
    forced = (os.getenv(f"CLASSIFIER_MODEL_{topic.upper()}") or "").strip().lower()
    recommendation = None if forced else read_recommendation(topic)
    key = forced or recommendation["recommended_model"]
    if key not in SUPPORTED_KEYS:
        raise ValueError(f"서비스에서 쓸 수 없는 모델입니다: {key!r} (가능: {', '.join(SUPPORTED_KEYS)})")
    best_dir = ARTIFACTS / topic / RECOMMEND_DATASET / key / "best_model"
    if not best_dir.is_dir():
        raise FileNotFoundError(f"best_model 폴더가 없습니다: {best_dir}")
    return key, recommendation["run_name"] if recommendation else None, best_dir


def shared_encoder():
    """CLM 인코더. 이미 떠 있는 서버는 그대로 쓰고, 없으면 직접 띄운다. 직접 띄운 서버는 close_classifiers()에서 끈다."""
    global _encoder
    if _encoder is None:
        add_path(JEV_SCRIPTS)
        from util.clm_encoder import ensure_encoder
        _encoder = _resources.enter_context(ensure_encoder())
    return _encoder


def build_backend(key: str, best_dir: Path, device):
    if key == "laya":
        return LayaBackend(best_dir, device)
    if key == "clm":
        return ClmBackend(best_dir, shared_encoder())
    return HfBackend(best_dir, device)


def load_topic(topic: str, device) -> ClassifierBundle:
    key, recommended_run, best_dir = choose_model(topic)
    info_path = best_dir / "best_info.json"
    run_name = json.loads(info_path.read_text(encoding="utf-8"))["best"]["run_name"] if info_path.exists() else None
    if recommended_run and run_name != recommended_run:
        logger.warning("[classifier] %s: 추천 때의 run(%s)과 지금 best_model의 run(%s)이 다릅니다. "
                       "추천 이후에 모델이 다시 학습된 것이니 recommendation.json을 다시 만드세요.",
                       topic, recommended_run, run_name)
    backend = build_backend(key, best_dir, device)
    return ClassifierBundle(topic=topic, model_key=key, run_name=run_name, backend=backend)


def load_classifier() -> dict:
    # 서버 시작 시 주제별 모델을 불러온다. 한 주제가 실패해도 서버는 뜨고, 그 주제는 요청 때 error를 돌려준다.
    device = resolve_device(os.getenv("CLASSIFIER_DEVICE") or None)
    bundles = {}
    for topic in TOPICS:
        try:
            bundles[topic] = load_topic(topic, device)
            bundle = bundles[topic]
            logger.info("[classifier] %s: %s (run %s) 로딩 완료, 라벨 %s", topic, bundle.model_key, bundle.run_name,
                        bundle.labels)
        except Exception as error:
            logger.exception("[classifier] %s 모델을 불러오지 못했습니다.", topic)
            bundles[topic] = ClassifierBundle(topic=topic, error=f"{type(error).__name__}: {error}")
    return bundles


def close_classifiers(bundles: dict | None = None) -> None:
    for bundle in (bundles or {}).values():
        if bundle is not None and bundle.backend is not None:
            bundle.backend.close()
    global _encoder
    _encoder = None
    _resources.close()      # 직접 띄운 인코더 서버 종료


def predict_text(text: str, bundle) -> dict:
    # 분류 결과. 앞의 세 키(status, label, confidence)가 API 응답 계약이고, 나머지는 추적·확인용이다.
    if bundle is None or bundle.error or bundle.backend is None:
        return {"status": "error", "label": None, "confidence": None,
                "error": getattr(bundle, "error", None) or "분류 모델이 준비되지 않았습니다."}
    try:
        result = bundle.backend.predict(text)
    except Exception as error:
        logger.exception("[classifier] %s 분류에 실패했습니다.", bundle.topic)
        return {"status": "error", "label": None, "confidence": None, "error": f"{type(error).__name__}: {error}"}
    return {
        "status": "success",
        "label": result["label"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "metrics": {"topic": bundle.topic, "model": bundle.model_key, "run_name": bundle.run_name,
                    "inference_ms": result["inference_ms"], "input_tokens": result["input_tokens"],
                    "truncated": result["truncated"], **result["extra"]},
    }
