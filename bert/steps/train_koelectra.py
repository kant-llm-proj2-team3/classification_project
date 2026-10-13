"""monologg/koelectra-small-v3-discriminator 분류 모델 학습 (bert/steps/train_koelectra.py).

모델 ID 와 Learning Rate 는 project_2/baseline_bert_config.py 에서 바꿉니다.
    PRETRAINED_MODELS["koelectra"], LEARNING_RATES["koelectra"]
결과: artifacts/<topic>/<dataset>/koelectra/<날짜 폴더>/ 에 저장하고, 지금까지의 best_model 보다 좋으면 best_model/ 을 교체

직접 실행 (프로젝트 루트 project_2 에서):
    uv run python bert/steps/train_koelectra.py                                   # ON 인 데이터셋 전부
    uv run python bert/steps/train_koelectra.py --topic inquiries --dataset basic
보통은 baseline/steps/run_menu.py 의 1번 메뉴에서 함께 실행됩니다.
"""
import sys
from pathlib import Path

# 설정/공통 함수는 baseline/steps 에 있다. (bert/steps 에서 두 단계 위가 project_2)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "baseline" / "steps"))

import common  # noqa: E402

MODEL_KEY = "koelectra"


def train(topic, dataset):
    """반환값: 실패한 run 이름 목록(모두 성공이면 빈 리스트)."""
    return common.train_model_runs(MODEL_KEY, topic, dataset)


def main():
    common.run_cli("monologg/koelectra-small-v3-discriminator 분류 모델 학습", train)


if __name__ == "__main__":
    main()
