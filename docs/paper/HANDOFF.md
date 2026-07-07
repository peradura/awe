# 논문 작성 핸드오프 (새 세션용)

**시작 프롬프트 예시**: "AWE 워크샵 논문 작성 이어가자 — docs/paper/draft.md 스켈레톤과
HANDOFF.md 읽고 진행."

## 상태 (2026-07-07)
- 스켈레톤 완성: `docs/paper/draft.md` (제목후보·초록·섹션구조·그림/표 계획·TODO).
- 방향 근거: codex+critic 교차리뷰 수렴 — "추가 실험 대신 논문 먼저, backfill은
  리뷰어 gap 2개만". 판정 기록: `docs/REVIEW.md`, `docs/exp_logs/LOG.md` 메모,
  `PROJECT.md` §7.

## 그라운드 트루스 (인용할 수치의 원천 — 문서 아닌 JSON 기준)
- 정준 bake-off: `results/bakeoff_reachp2_s{0..9}.json` (Part 4c)
- 외부 전이: `results/mqar_seed*.json`, `results/mqarhop_seed*.json` (Part 5)
- 역사/견고성: `results/reachp3_seed*.json` (4a), sweep 로그 (4b)
- 서술 원천: `docs/RESULTS.md` (Part 4a/4b/4c/5), `PROJECT.md` §4

## 절대 규칙 (이미 리뷰에서 확정된 것 — 어기면 안 됨)
1. 두 recon 정의(state-mismatch vs decodability)를 절대 합치지 말 것.
2. 4a/4b/4c 표를 한 표로 합치지 말 것 — 인용은 4c(정준)로.
3. "one signal, two knobs"를 살아있는 주장처럼 쓰지 말 것 — halting 축에서
   검정·기각된 사실로 서술 (이게 논문의 정직성 자산).
4. write-magnitude probe는 본문 주장 금지 (2중 confound — 후속 논문 시드).
5. 모든 수치 mean±std·10 seeds·held-out tau 표기, 예외는 명시.

## 남은 작업 순서
1. draft.md 섹션 살 붙이기 (Intro→4장 결과 순서 권장; 수치는 JSON에서 재검증하며)
2. Fig 1 multi-seed band 재생성 스크립트 (bakeoff JSONs의 curve 필드 사용)
3. backfill 실험 2개 (GPU는 완전 유휴 시 + 사용자 승인): depth_sanity 다시드(CPU 가능),
   표준 zoology-MQAR 앵커 1개(설계 먼저)
4. LaTeX 포팅은 내용 안정화 후
