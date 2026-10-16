# 분류 모델 비교 실험 (baseline vs BERT / RoBERTa / KoELECTRA)

TF-IDF + 로지스틱 회귀(baseline)와 사전학습 모델 3종을 같은 데이터로 학습하고,
결과로 서비스에 쓸 분류기 후보를 추천합니다. (laya, CLM, jev 같은 외부 모델 결과도 같은 표에 합칩니다)

## 파일 구성

| 파일 | 역할 |
|---|---|
| `baseline_bert_config.py` (프로젝트 루트) | **모든 설정**: 데이터셋 ON/OFF, 모델별 Learning Rate, 에폭·배치·조기 종료, 결과 경로 |
| `baseline/steps/run_menu.py` | **입력 메뉴 (보통 이것만 실행)** |
| `baseline/steps/common.py` | 데이터 검사, 학습 루프, best_model 비교·교체, 모델 추천, test 평가, 초기화 |
| `baseline/steps/train_tfidf.py` | TF-IDF + 로지스틱 회귀 학습 |
| `baseline/steps/check_lengths.py` | CSV 파일별 입력 길이(토큰 수) 점검 (메뉴 5번) |
| `bert/steps/train_bert.py`, `train_roberta.py`, `train_koelectra.py` | 모델별 학습 파일 |
| `baseline/steps/make_charts.py`, `result_files.py` | 그래프(메뉴 7번), `main.json`/`sub.json`/예측 CSV 생성 (메뉴 6번) |

## 실행

프로젝트 루트에서 실행합니다.

```bash
uv run python baseline/steps/run_menu.py
```

| 메뉴 | 동작 |
|---|---|
| 1 학습 | baseline + bert + roberta + koelectra 학습. **누를 때마다 항상 새로 학습**하고, 그 모델의 best_model 보다 좋을 때만 best_model 을 교체 |
| 2 초기화 | 학습 결과(`artifacts`)만 삭제. 범위(전체/모델별) 선택, `yes` 입력 필요. CSV 와 `logs/` 는 삭제되지 않음 |
| 3 추천 | 각 모델 best_model 로 1개 추천 (`recommendation.json`). 현재 규칙은 아래 "모델 추천 규칙" 참고 |
| 4 test 평가 | 추천 모델의 best_model 을 test 로 **1회만** 평가. baseline 의 test 점수도 같이 기록 |
| 5 길이 점검 | CSV 파일별로 192/512토큰을 넘는 행 수 확인 (`artifacts/input_length_report.csv`) |
| 6 결과 파일 보강 | 이미 있는 학습 결과에 `main.json` / `sub.json` / 문장별 예측 파일 생성 (다시 학습하지 않음) |
| 7 그래프 | 모든 `best_model/main.json`으로 비교 그래프 PNG 생성 |
| 8 한 번에 마무리 | 6 → 3 → 4(4개 모델) → 7 순서로 실행 |

모델 하나만 따로 돌리려면 다음처럼 실행할 수 있습니다.

```bash
uv run python bert/steps/train_bert.py --topic inquiries --dataset basic
uv run python baseline/steps/train_tfidf.py --topic inquiries --dataset basic
```

## 결과 폴더 구조

```
artifacts/
├── inquiries/  documents/                       주제
│   ├── basic/  basic_merge/  advanced/  advanced_merge/  total_merge/   데이터셋
│   │   ├── baseline/  bert/  roberta/  koelectra/         모델
│   │   │   ├── best_model/        지금까지 가장 좋았던 결과 (가중치 포함)
│   │   │   ├── 20261012_195030/   학습 1회 = 폴더 1개 (날짜_시각, 점수·로그만)
│   │   │   └── 20261012_210544/
│   │   ├── recommendation.json                  (3번 추천) 추천 모델과 이유
│   │   └── (모델 간 비교표는 화면에만 출력합니다)
└── input_length_report.csv                       (5번) 입력 길이 점검
```

`best_model/` 안:

| 파일 | 내용 |
|---|---|
| `model.safetensors`, `config.json`, 토크나이저 파일 | 사전학습 모델 가중치 (baseline 은 `baseline.joblib`). **Git 에 올라가지 않음** |
| `best_info.json` | **어느 날짜 run 이 best 인지**, 점수, 이전 best 와의 비교, 전체 run 순위표 |
| `run_config.json` | 그 run 의 학습 조건 (에폭, 배치, 입력 길이, LR, 잘린 행 수 등) |
| `training_summary.json`, `validation_metrics.json`, `validation_errors.csv`, `epochs.jsonl`, `train.log` | 점수, 라벨별 지표·혼동행렬, 틀린 문장, 에폭별 기록, 로그 |
| `test_metrics.json`, `test_errors.csv` | (4번을 실행한 뒤) test 최종 평가 |

날짜 폴더에는 `best_model` 과 같은 파일이 **가중치 없이** 들어 있습니다.
Hugging Face 가 모델 설정을 `config.json` 으로 저장하기 때문에, 우리의 학습 조건 기록은 `run_config.json` 입니다.

## 모델 추천 규칙 (현재 코드 기준)

- 추천 순위는 각 모델 `best_model` 의 **test Macro-F1** 이 높은 순이고, 동점이면 validation loss 가 낮은 쪽, 그다음 학습 시간이 짧은 쪽입니다. (`common.py` 의 `recommend_model`)
- 따라서 추천 전에 test 평가(메뉴 4번)가 끝나 있어야 하고, **이 test 점수는 모델 선택에 쓰이므로** 처음 보는 데이터에서의 성능보다 높게 나올 수 있다는 점이 `recommendation.json` 의 `notes` 에 기록됩니다.
- baseline 과 `RECOMMEND_REFERENCE_ONLY`(jev) 는 후보에서 제외되고 참고용으로만 표시됩니다.
- 서비스(FastAPI)는 `advanced_merge` 의 `recommendation.json` 을 읽어 서비스 모델을 정합니다.
- 아래 "4. test 평가" 와 "평가 규칙" 에 적힌 validation 중심 설명은 학습·best_model 교체 기준입니다.

## best_model 규칙

- 학습이 끝날 때마다 새 날짜 폴더가 생기고, 그 모델의 best_model 과 비교합니다.
- **validation Macro-F1 이 더 높으면 교체**, 동점이면 validation loss 가 낮은 쪽이 이깁니다. 같거나 나쁘면 유지합니다.
- 교체되면 가중치가 `best_model` 로 옮겨지고 **이전 best 의 가중치는 사라집니다**. (점수·로그는 그 날짜 폴더에 남음)
- 교체되지 않으면 이번 run 의 가중치는 지웁니다. (점수·로그만 날짜 폴더에 남음)
- 이미 test 평가를 마친 best 가 교체되면 경고를 출력하고, 이전 test 결과는 그 run 의 날짜 폴더로 옮겨 보관합니다.
- 학습 중 실패하거나 중단(Ctrl+C)한 run 은 폴더가 남지 않습니다. (끊긴 폴더가 있으면 다음 학습 때 지움)

## 자주 바꾸는 설정 (`baseline_bert_config.py`)

- **데이터셋 켜기/끄기**: `DATASET_SWITCH` 에서 해당 줄을 `True`/`False` 로 변경 (데이터 폴더가 비어 있으면 자동으로 건너뜀)
- **데이터셋 구성**: `basic`, `advanced`(원본), `basic_merge`, `advanced_merge`(원본 + 추가본), `total_merge`(basic_merge + advanced_merge).
  폴더 위치는 `DATASET_FOLDERS`. 예전 이름 `basic_plus`, `advanced_plus` 는 `basic_merge`, `advanced_merge` 로 바뀌었습니다.
- **모델 추천에서 빼기**: `RECOMMEND_EXCLUDED` (현재 기본값 `()` = 모든 데이터셋을 추천). `total_merge` 는 서버에 줄 최종 모델을 만드는 데이터셋이라 학습, test 평가, 추천, 그래프에 모두 포함됩니다.
  예전처럼 빼려면 `("total_merge",)` 를 적습니다. 여기 적은 데이터셋은 4개 모델을 모두 test 평가하고 공식 모델 표시는 하지 않으며 그래프에서도 빠집니다(`--with-total-merge` 로 포함).
- **Learning Rate**: `LEARNING_RATES` 에 모델별 숫자 하나. `[2e-5, 5e-5]` 처럼 리스트로 적으면 값마다 run 이 하나씩 돌고 best 가 선택됨(튜닝)
- **배치**: `batch_size`(한 번에 넣는 문장 수) x `gradient_accumulation_steps`(몇 묶음마다 가중치를 고칠지) = 실질 배치. 기본 4 x 2 = 8
- **에폭·중단 조건**: `epochs`(최대 15), `early_stopping_patience`(Macro-F1 이 N에폭 연속 안 오르면 중단, 기본 3, 0 이면 끔), `target_macro_f1`(숫자를 적으면 그 점수에 닿는 순간 중단, 기본 None=끔)
- **점수 기준 조기 종료**: `stop_when_scores_reached`(켜기/끄기), `stop_min_accuracy`(기본 1.0=만점), `stop_min_macro_f1`(기본 1.0),
  `stop_max_val_loss`(기본 0.01), `stop_min_epochs`(기본 3). 세 조건이 **한 에폭에서 모두** 만족되면 그 에폭에서 멈춥니다.
  값을 `None` 으로 두면 그 조건은 보지 않고, 종료 사유는 로그에 실제 값과 함께 남습니다. (예: "점수 기준 도달: 정답 120/120(만점), Macro-F1 1.0000, val_loss 0.0085 <= 0.01")
- **입력 길이·자르는 방식**: `max_length`(최대 512), `truncation`(`"right"` 앞쪽만 남김 / `"head_tail"` 앞뒤를 남김), `head_tokens`.
  advanced 는 정답 구간("새로 접수한 문의" / 문서는 "이번에 전달하는 본문")이 글의 앞에도, 맨 뒤에도 있어서 기본값은 `512` + `head_tail`(앞 256 + 뒤 254)입니다.
  학습 로그에 "max_length 를 넘어 잘린 행" 수가 출력되고, test 평가는 best_model 의 `run_config.json` 에 기록된 방식 그대로 입력합니다.

설정을 바꿔도 이전 결과는 그대로 남고, 새 학습 결과는 best_model 과 비교될 뿐입니다. 이전 결과를 지우려면 2번(초기화)을 쓰세요.

## 결과 파일과 그래프 (메뉴 6, 7번)

학습할 때마다 날짜 폴더와 `best_model` 에 아래 파일이 함께 생깁니다. 이미 있는 결과에는 메뉴 6번이 (다시 학습하지 않고) 만듭니다.

| 파일 | 내용 |
|---|---|
| `main.json` | **핵심 결과** (모든 run 이 같은 키): 모델·데이터셋, 학습 조건 요약, validation/test 의 "몇 문제 중 몇 개 정답"·accuracy·Macro-F1·라벨별 정답·혼동행렬, 비용(학습·추론 시간, GPU 메모리, 파라미터 수, 모델 크기), 에폭별 곡선 |
| `sub.json` | 부수 정보: 시드, 입력 길이·자르는 방식·잘린 행 수, 중단 사유, 세부 하이퍼파라미터, 실행 환경, 라벨별 precision·recall, 에폭 상세 |
| `validation_predictions.csv` | validation **모든 문장**의 `id`, `actual`, `predicted`, `correct`, `confidence`, 라벨별 확률 |
| `test_predictions.csv` | test 모든 문장의 예측 (메뉴 4번을 실행한 이후) |

- `main.json` 의 `test` 칸은 메뉴 4번 전에는 `null` 입니다. `schema` 필드는 파일 형식의 버전 표시입니다.
- 메뉴 7번(그래프)은 모든 데이터셋 x 모델의 `best_model/main.json` **만** 읽어 PNG 그래프를 `artifacts/viz/charts/` 에 만듭니다. (CSV 표는 만들지 않습니다)
  `01_heatmap_test`(데이터셋 x 모델 test 정확도), `02_heatmap_validation`, `03_baseline_vs_official`, `04_learning_curves`, `05_cost_vs_accuracy`,
  `06_confusion_official`. 직접 실행할 수도 있습니다: `uv run python baseline/steps/make_charts.py` (처음 한 번 `uv add matplotlib`, 서버용 total_merge 를 넣으려면 `--with-total-merge`).
- 팀원과 비교할 때는 문장별 예측 CSV(`id`, `actual`, `predicted`)를 같이 주고받으면, 같은 코드로 지표를 다시 계산하고
  같은 문장에서 누가 맞고 틀렸는지도 비교할 수 있습니다.

## test 평가 (메뉴 4번)

| 방식 | 평가하는 것 | 용도 |
|---|---|---|
| 1) 추천 모델 1개 | 추천 모델의 `best_model` + baseline | 최종(공식) 점수 |
| 2) 4개 모델 모두 | baseline + bert + roberta + koelectra 각 `best_model` | 비교용 |

- 최종 모델은 어느 쪽이든 **validation 으로 정한 추천 모델**입니다. 추천 모델의 점수만 `official`(공식), 나머지는 참고용으로
  `test_metrics.json` 과 `main.json` 의 `test_meta` 에 표시되고, test 점수를 보고 모델을 다시 고르면 안 됩니다.
- 모델은 각자 한 번만 평가합니다. (이미 평가한 모델은 건너뜀. baseline 은 비교 기준선이라 항상 다시 계산해도 같은 값)
- 각 모델 `best_model` 에 `test_metrics.json`, `test_errors.csv`, `test_predictions.csv` 가 저장되고,
  데이터셋별 모델 비교표는 화면에만 출력되고 파일로는 저장되지 않습니다.

## 한 번에 마무리 (메뉴 8번)

현재 best_model 로 **빠진 결과 파일 보강 -> 모델 추천 -> 4개 모델 test 평가 -> 그래프 만들기**를 한 번에 실행합니다. (다시 학습하지 않음)
학습이 모두 끝난 뒤에 실행하세요. test 는 모델마다 한 번만 평가되고, 공식 모델은 validation 으로 정한 추천 모델입니다.

학습(1번)이 끝날 때마다 새 run 이 best 를 못 이겼을 때 남는 옛 best_model 에 main.json / sub.json / 문장별 예측이 없으면
자동으로 채웁니다. (그 best 자신의 기록으로 main/sub 를 정리하고, 예측은 저장된 가중치로 validation 을 한 번 더 예측해 만들며 다른 run 의 파일을 복사하지 않음)

## 외부 모델 (laya, jev, clm) 결과 합치기

다른 팀원이 직접 학습해서 결과만 넘겨 주는 모델을 우리 모델과 같은 표에 넣습니다. 우리 코드는 이 모델을 학습하거나 test 를 직접 평가하지 않고,
결과 파일만 읽습니다.

- **켜고 끄기**: `baseline_bert_config.py` 의 `EXTERNAL_MODELS` (현재 `{"laya": True, "jev": True, "clm": True}`, 끄려면 해당 모델을 `False` 로). `False` 면 무시합니다.
- **폴더 위치**: `artifacts/<주제>/<데이터셋>/<모델 소문자 이름>/best_model/main.json` (예: `artifacts/documents/advanced/laya/best_model/main.json`).
  폴더가 없는 데이터셋은 오류 없이 건너뜁니다.
- **모델 추천(3번·8번)**: `main.json` 의 **validation** 결과로 우리 모델과 같이 순위를 매깁니다. (`main.json` 이 없는 예전 결과만 `best_info.json` 으로 대신 읽음) validation 결과가 없거나
  validation 문장 수가 우리와 다르면 비교에서 빠집니다. loss 가 비어 있으면 동점일 때 뒤로 보냅니다.
- **test 비교(8번)**: 외부 모델의 test 점수는 팀원이 넘긴 `main.json` 의 `test` 칸 값을 그대로 씁니다.
  `test` 칸이 비어 있으면 화면에 알리고 표에는 빈칸으로 남습니다. 공식 여부는 추천 결과(`recommendation.json`)로 정합니다.
- **그래프(7번·8번)**: 같은 그래프에 함께 들어갑니다.
- **참고용으로만 넣기**: `RECOMMEND_REFERENCE_ONLY` (기본 `("jev",)`)에 적힌 외부 모델은 baseline 처럼 모델 추천(최종 모델 선정)에서 빼고
  표·그래프에 참고용으로만 들어갑니다. validation 칸이 없어도 되고 test 결과만 있어도 됩니다.
- **초기화(2번)**: 우리 모델(baseline, bert, roberta, koelectra)만 지웁니다. 외부 모델 폴더는 지우지 않습니다.
- 형식이 다르면(모델 이름, dataset_id, validation·test 문장 수, 라벨 순서) 7번·8번 실행 시 `[주의]` 경고가 나옵니다.

## 로그

- `logs/<종류>_<실행시각>.log` : 학습(train) · 추천(recommend) · test · 길이점검(lengths)을 실행할 때 화면에 찍힌 내용 전체.
  맨 위에 그때의 설정(에폭, 배치, 중단 조건, 입력 길이, LR 등)이 기록되며, 초기화(2번)를 해도 지워지지 않습니다.
- 각 run 폴더의 `train.log` : 그 run 의 설정 + 에폭별 로그 + best_model 갱신/유지 판정.
- `.gitignore` 에 `*.log` 와 `artifacts/` 가 들어 있어 둘 다 Git 에는 올라가지 않습니다.
  제출용 파일은 `.gitignore` 를 조정하거나 별도 폴더에 복사하세요.

## 평가 규칙

- 학습은 `train` 만 사용합니다. 모델·epoch 선택과 best_model 교체는 `validation` Macro-F1 만 봅니다
  (동점이면 validation loss → 학습 시간 순).
- `test` 는 모델 추천이 끝난 뒤 4번 메뉴에서만 읽으며, 이 점수를 보고 모델·설정을 다시 고르지 않습니다.
- 데이터 검사: id 중복(분할 안팎), 분할 간 group_id·문장 겹침은 에러로 멈춥니다.
  같은 group_id 안에 다른 라벨이 섞인 경우는 경고만 합니다 (advanced 는 정상, basic 은 확인 필요).
- 라벨 순서는 `labels.json` 순서이며 `baseline_bert_config.py` 의 `LABELS` 와 다르면 멈춥니다.
- basic, advanced, merge, total_merge 는 validation·test 가 서로 달라서 점수를 직접 비교하기 어렵습니다.
  비교표에 평가 세트의 행 수를 함께 적어 두세요.
