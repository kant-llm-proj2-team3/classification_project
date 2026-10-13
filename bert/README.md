## bert/steps



| 파일 | 모델 | 설정 키 |
|---|---|---|
| `train_bert.py` | klue/bert-base | `bert` |
| `train_roberta.py` | klue/roberta-base | `roberta` |
| `train_koelectra.py` | monologg/koelectra-small-v3-discriminator | `koelectra` |


- 세 파일은 구조가 같고 모델 이름만 다릅니다. 실제 학습·평가·저장 로직은 `baseline/steps/common.py`의 `train_model_runs` 입니다.
- 모델 ID와 학습률은 `baseline_bert_config.py`에서 변경 가능합니다.
- 결과는 `artifacts/<topic>/<dataset>/<모델>/<날짜 폴더>/`에 저장되고, 기존 `best_model`보다 좋을 때만 `best_model/`이 교체됩니다.
- 보통은 `baseline/steps/run_menu.py`의 1번 메뉴에서 baseline과 함께 실행됩니다.
- 한 모델만 따로 실행할 수도 있습니다.

```bash
uv run python bert/steps/train_roberta.py --topic inquiries --dataset basic
```