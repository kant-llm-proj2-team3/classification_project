# 부하 테스트 (Locust)

`/chat` 엔드포인트에 소규모 부하를 주고, 분류와 생성이 모두 성공해야 성공으로 집계하는 테스트입니다. 정답률 측정용이 아니라 응답 시간·오류·서버 추적 기록을 보기 위한 용도입니다.

## 파일

| 파일 | 설명 |
|---|---|
| `locust/locust.py` | Locust 시나리오. 질문 CSV를 읽어 `/chat`을 호출하고, 요청마다 고유한 추적 ID를 붙임. 응답 본문과 API 키는 결과에 남기지 않음 |
| `locust/advanced_merge_100.csv` | 질문 입력 (`dataset`, `text`, `label`, `id`, `group_id`, `source_split`, `source`, `input_chars`). `advanced_merge` 기반 |
| `scripts/run_load_report.py` | 여러 단계(사용자 수)를 연속 실행하고 Locust 측정값을 서버 추적 DB(`logs/tracking.db`)와 합쳐 보고서를 만듦 |

## 안전 장치
- 한 프로세스에서 보내는 새 `/chat` 요청은 기본 최대 40건(`--max-requests`)입니다. 분산 실행(master/worker)은 막혀 있고, 실패한 요청은 자동 재시도하지 않습니다.
- 사용자 대기 시간은 기본 5~15초(`--think-min`, `--think-max`)입니다.
- 실제 OpenAI 호출이 발생하므로 비용이 듭니다. 실행 전에 서버가 떠 있고 `/health`의 분류 모델 상태가 정상이어야 합니다.

## 실행

```powershell
# 서버를 먼저 띄운 뒤 (fastapi/README.md 참고)
uv run python scripts/run_load_report.py --output-dir <결과 폴더> --host http://127.0.0.1:8000
uv run python scripts/run_load_report.py --output-dir <결과 폴더> --smoke-only   # 스모크 단계만
```

`--tracking-db`(기본 `logs/tracking.db`)는 읽기 전용으로 열며, 실행 결과로 `locust/result_*.csv`, `locust/report_*.html` 같은 산출물이 생길 수 있습니다. (두 패턴은 `.gitignore`에 포함)
Locust를 직접 실행하려면 `uv run locust -f locust/locust.py --host http://127.0.0.1:8000 --headless -u <사용자 수> -r <초당 증가>`를 사용합니다.

## 해석할 때 주의
- 단계당 4~12건 수준의 탐색 결과이므로 P95/P99와 수용 가능 인원을 확정하기에는 표본이 부족합니다.
- 프론트 UI, WebSocket, 수천 명 규모 동시 접속은 검증하지 않았습니다.
- 분류의 "비추론 시간"에는 토큰화·스레드·락 대기가 섞여 있습니다.
