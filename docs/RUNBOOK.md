# RUNBOOK — 공유 GPU 서버에서 실험 돌리기

랩 공유 서버에서 다른 사용자에게 피해 없이 실험 큐를 돌리기 위한 절차.
(모델은 0.2~0.9M 파라미터라 GPU 1장 일부만 써도 충분하다.)

## 1. 준비 (최초 1회)

```bash
git clone https://github.com/peradura/awe.git awe && cd awe
git checkout claude/project-review-discussion-b0r8j3

# 요즘 서버 파이썬은 PEP 668(externally-managed)이라 venv가 필요하다:
python3 -m venv .venv && source .venv/bin/activate
pip install torch matplotlib
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

이후 세션에서는 `cd awe && source .venv/bin/activate`만 하면 된다.
tmux가 없으면 `nohup <명령> > 로그 2>&1 &`로 대체 (아래 예시 참조).

## 2. 예절 러너 — GPU가 빌 때까지 기다렸다가 자동 실행

```bash
# 10분(600s)마다 nvidia-smi로 적재 확인, 유휴 GPU(메모리<=1GB, util<=10%)가
# 생기면 그 GPU 하나만 CUDA_VISIBLE_DEVICES로 점유하고 큐 실행:
nohup bash scripts/gpu_watch_run.sh > gpu_watch.out 2>&1 &
tail -f gpu_watch.out

# 특정 GPU를 기다리려면:      bash scripts/gpu_watch_run.sh 1
# 폴링 주기/임계값 조정:      POLL_SEC=600 MEM_MAX_MB=1000 UTIL_MAX=10 bash scripts/gpu_watch_run.sh
# 큐 구성 변경(순서대로 실행): QUEUE="reachp2 sweep depth_sanity" bash scripts/gpu_watch_run.sh
```

기본 큐: `reachp2`(관문 실험, 8k스텝) → `sweep`(seed 0–4 × {rule, reachp,
bakeoff×2} + 자동 집계).

## 3. 개별 실행 (GPU를 이미 확보했다면)

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID        # CUDA 인덱스 = nvidia-smi 인덱스
export CUDA_VISIBLE_DEVICES=<idx>          # 반드시 GPU 하나만 지정
PYTHONPATH=src python3 -m awe.experiments.ablation_reachp2 --steps 8000 --fig results/reachp2_curve.png |& tee results/reachp2_run.log
SEEDS="0 1 2 3 4" bash scripts/sweep.sh    # 다중 시드 + bake-off + mean±std 집계
python3 scripts/aggregate.py results       # 집계만 다시 보기
```

sweep은 ablation이 저장한 체크포인트(`results/ckpt_*.pt`)를 bake-off가 재사용해
같은 모델을 두 번 학습하지 않으며, 한 단계가 죽어도 나머지 큐는 계속 돈다
(실패는 `results/sweep_failures.log`에 기록).

## 4. 끝나면 반드시 커밋

```bash
git add -f results/*.log results/*.png     # .gitignore가 로그/그림을 막으므로 -f
git commit -m "Results: reachp2 + multi-seed sweep + bake-off (GPU)"
git push -u origin claude/project-review-discussion-b0r8j3
```

## 5. 결과 해석 가이드

- **reachp2**: persist가 K≥2 answerable에서 ~70%+ 나오면 base-learner 병목 해소
  → bake-off 판정이 유효해짐. 여전히 낮으면 컨트롤러 논의 전에 학습부터.
- **bake-off** (`bakeoff_{rule,reachp}_s*.log`): 각 신호(ent/recon/rnorm/dstate/dent)의
  acc·steps·corr과 실패 분해(early_right/premature/wrong_anyway/full_depth, 합=1;
  `pm|halt`=조기 halt 중 premature 비율).
  - **판정은 단일 점이 아니라 `curve` 줄(신호별 tau-스윕 eval 곡선)에서** — 같은
    avg-steps에서 acc를 비교해야 matched-compute 비교(kill criterion)가 된다.
  - `recon`이 `ent`와 대등하면 → 논지에 유리. 단 `recon`은 문자 그대로의 TTT write
    loss가 아니라 그 자기지도 **대체물**(read decodability — probe의 정답 v를 아는
    write loss는 halting 신호가 될 수 없음). 판정 문구도 그렇게 써야 함.
  - `shuf_b`(쿼리-위치별 스텝 분포 보존 널)와 차이가 없으면 그 신호는 per-example
    정보를 안 쓰는 것. `shuf_g`만 이기는 건 위치 스케줄 효과일 수 있음.
  - `premature`가 큰 신호는 확신-오답 조기 종료 문제(Part 3의 −9.6pp 원인) 보유.
  - `dent`는 구조상 t=0에 halt 불가(+1스텝 페널티) — steps 비교 시 감안.
  - bake-off의 `ent`는 ablation의 `both`와 tau 선택 방식이 달라(median vs
    slack-기반) 수치가 정확히 일치하지 않는 것이 정상.
- **중단 기준**: PROJECT.md §7 kill criterion 참조.

## 주의

- `nvidia-smi`가 유휴로 보여도 남이 곧 쓸 예약이 있을 수 있음 — 랩 규약(예약
  보드/슬랙)이 있으면 그것이 우선.
- 러너는 GPU **하나만** 점유한다. 멀티-GPU 학습은 없다.
- 이 컨테이너/세션 환경에는 GPU가 없다 — 이 러너는 랩 서버 전용.
