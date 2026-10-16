# artifacts

학습·평가 산출물 디렉토리입니다. 대부분 코드가 자동으로 만들며 직접 수정하지 않습니다. 가중치 파일(`*.safetensors`, `*.joblib`, `*.pt`)과 `*.log`는 Git에 올라가지 않습니다.

```text
artifacts/
├── <topic>/                      documents, inquiries
│   └── <dataset>/                basic, basic_merge, advanced, advanced_merge, total_merge
│       ├── <model>/              baseline, bert, roberta, koelectra, laya, laya_zeroshot, clm, clm_zeroshot, jev
│       │   ├── best_model/       지금까지 가장 좋았던 결과 (main.json, sub.json, 예측, 오답, 가중치)
│       │   └── <날짜_시각>/       학습·평가 1회 = 폴더 1개 (점수·로그, 가중치 없음)
│       └── recommendation.json   모델 추천 결과 (추천 모델, 선정 규칙, 이유)
├── viz/charts/                   비교 그래프 PNG (히트맵, baseline 대비, 학습 곡선, 비용 대비 정확도)
├── generation_compare/<이름>/    생성 모델(OpenAI vs Ollama) 비교와 사람 채점 결과
├── serving_check/                (서빙 경로 점검을 실행하면 생성)
└── input_length_report.csv       (baseline 메뉴 5번 실행 시 생성) 입력 길이 점검
```

## 만든 코드
| 산출물 | 만드는 코드 | 문서 |
|---|---|---|
| baseline / bert / roberta / koelectra, recommendation.json, viz | `baseline/steps/run_menu.py` | [baseline](../baseline/README.md) |
| laya, clm, jev (+ zeroshot) | `jev/scripts/step04~06` | [jev](../jev/README.md) |
| generation_compare | `fastapi/evaluation/compare.py`, `report.py` | [fastapi](../fastapi/README.md) |

## 서비스와의 관계
FastAPI는 시작할 때 `artifacts/<토픽>/advanced_merge/recommendation.json`을 읽어 같은 폴더의 `<추천 모델>/best_model`을 불러옵니다. 따라서 이 파일과 해당 `best_model`(가중치 포함)이 있어야 서비스를 띄울 수 있습니다.
현재 추천: documents → `bert`, inquiries → `laya`. (`total_merge`에서는 두 토픽 모두 `roberta`)

## 모델 추천 규칙
- 학습·best_model 교체는 validation Macro-F1(동점이면 validation loss)로 정합니다.
- 추천(`recommendation.json`)은 각 모델 best_model의 **test** Macro-F1(동점이면 validation loss → 학습 시간) 순입니다. test 점수가 모델 선택에 쓰였으므로 처음 보는 데이터에서의 성능보다 높게 나올 수 있습니다. baseline과 jev는 참고용으로만 비교하고 후보에서 제외합니다.

## 결과 요약 (`advanced_merge`, test Macro-F1)
| 모델 | documents | inquiries |
|---|---:|---:|
| baseline | 0.875 | 0.809 |
| bert | 1.000 | 0.973 |
| roberta | 0.991 | 0.982 |
| koelectra | 0.991 | 1.000 |
| laya | 0.973 | 1.000 |
| clm | 0.928 | 0.910 |
| jev (API, 참고용) | 0.919 | 0.982 |
| laya_zeroshot | 0.488 | 0.533 |
| clm_zeroshot | 0.379 | 0.357 |

## 생성 모델 비교 (`generation_compare/compare_20261015_113659`)
test에서 토픽·종류(basic, basic_add, advanced, advanced_add)마다 10건을 뽑아 openai와 ollama를 호출(각 80건)하고, 최선영이 0~2점 4개 항목(정책 정확, 필수 항목, 근거 없는 주장 없음, 명료성)으로 채점했습니다.

| 모델 | 오류율 | 형식 유효율 | 응답 시간 평균(초) | 총점 평균(0~8) | 없는 사실 0점 |
|---|---:|---:|---:|---:|---:|
| openai | 0.000 | 1.000 | 2.56 | 7.41 | 0 |
| ollama | 0.013 | 0.988 | 2.98 | 6.78 | 5 |

같은 입력 80건 중 openai 우세 32, 동점 44, ollama 우세 4입니다. 표본이 작으므로 `report.md`의 주의 사항을 함께 보세요.
`report.md`가 `response_edited` 경고를 낸 것은 채점 파일의 답변 원문이 변경된 것으로 감지됐기 때문이니, 해석 시 확인이 필요합니다.
