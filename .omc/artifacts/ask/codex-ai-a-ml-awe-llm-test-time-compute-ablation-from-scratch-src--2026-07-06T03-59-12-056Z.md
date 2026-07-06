# codex advisor artifact

- Provider: codex
- Exit code: 0
- Created at: 2026-07-06T03:59:12.057Z

## Original task

AI 연구 학생의 여러 프로젝트에 쓸 "연구 레포 구조 템플릿"을 유형별로 최적화하고 싶어. 비판적으로 검토하고 구체적 템플릿을 제안해줘.

## 배경
같은 사람이 성격이 다른 연구 프로젝트를 여럿 함. 구조를 하나로 강제하지 말고 유형별로 최적화하되, 공통 규약은 공유하고 싶음.

## 유형 A — ML 파일럿/방법 개발 (예: AWE, LLM test-time compute)
특징: 실험 스크립트 많음, 모델·데이터생성기·ablation 반복, 빠른 반복, from-scratch 소형모델.
현재 구조(내가 이미 적용):
```
src/<pkg>/{datasets,models,experiments}/   # 코드
docs/{proposal.md, RESULTS.md, exp_logs/LOG.md}
PROJECT.md, LOGGING.md, README.md, pyproject.toml(install -e), results/
```
실행: python -m <pkg>.experiments.<name>. LOG.md는 날짜|실험|컨셉|결과(판정)|산출물 인덱스.

## 유형 B — XAI/분석+논문 (예: PEMFC SEM segmentation+XAI)
특징: 데이터 분석·시각화·XAI 해석 중심, notebook 많음, 최종 산출물이 논문/figure. 다른 에이전트가 만든 구조:
```
data/ models/ xai/ analysis/ paper/  + README.md, ROADMAP.md, .gitignore
```

## 질문
(a) 유형 A 구조는 적절한가? 개선점? (특히 data/ 대용량·notebook·config·seed/재현성·results vs artifacts 분리)
(b) 유형 B(XAI/분석/논문)의 최적 구조는? 위 data/models/xai/analysis/paper가 좋은가, 아니면 더 나은 레이아웃? notebooks/ configs/ reports/ 등 포함 여부.
(c) 두 유형이 공유해야 할 공통 규약(README/PROJECT/LOGGING/exp_logs/.gitignore/데이터 버전관리/재현성)은?
(d) 데이터·큰 파일·모델 체크포인트는 git에서 어떻게(예: .gitignore + DVC/git-lfs/외부저장 포인터)?
각 유형별로 바로 쓸 수 있는 디렉토리 트리 + 공통 규약 체크리스트로. 간결하고 구체적으로.

## Final prompt

AI 연구 학생의 여러 프로젝트에 쓸 "연구 레포 구조 템플릿"을 유형별로 최적화하고 싶어. 비판적으로 검토하고 구체적 템플릿을 제안해줘.

## 배경
같은 사람이 성격이 다른 연구 프로젝트를 여럿 함. 구조를 하나로 강제하지 말고 유형별로 최적화하되, 공통 규약은 공유하고 싶음.

## 유형 A — ML 파일럿/방법 개발 (예: AWE, LLM test-time compute)
특징: 실험 스크립트 많음, 모델·데이터생성기·ablation 반복, 빠른 반복, from-scratch 소형모델.
현재 구조(내가 이미 적용):
```
src/<pkg>/{datasets,models,experiments}/   # 코드
docs/{proposal.md, RESULTS.md, exp_logs/LOG.md}
PROJECT.md, LOGGING.md, README.md, pyproject.toml(install -e), results/
```
실행: python -m <pkg>.experiments.<name>. LOG.md는 날짜|실험|컨셉|결과(판정)|산출물 인덱스.

## 유형 B — XAI/분석+논문 (예: PEMFC SEM segmentation+XAI)
특징: 데이터 분석·시각화·XAI 해석 중심, notebook 많음, 최종 산출물이 논문/figure. 다른 에이전트가 만든 구조:
```
data/ models/ xai/ analysis/ paper/  + README.md, ROADMAP.md, .gitignore
```

## 질문
(a) 유형 A 구조는 적절한가? 개선점? (특히 data/ 대용량·notebook·config·seed/재현성·results vs artifacts 분리)
(b) 유형 B(XAI/분석/논문)의 최적 구조는? 위 data/models/xai/analysis/paper가 좋은가, 아니면 더 나은 레이아웃? notebooks/ configs/ reports/ 등 포함 여부.
(c) 두 유형이 공유해야 할 공통 규약(README/PROJECT/LOGGING/exp_logs/.gitignore/데이터 버전관리/재현성)은?
(d) 데이터·큰 파일·모델 체크포인트는 git에서 어떻게(예: .gitignore + DVC/git-lfs/외부저장 포인터)?
각 유형별로 바로 쓸 수 있는 디렉토리 트리 + 공통 규약 체크리스트로. 간결하고 구체적으로.

## Raw output

```text
아래처럼 가는 게 가장 실용적입니다. 핵심은 **A는 “실험 엔진 중심”**, **B는 “분석 산출물/논문 중심”**으로 분리하고, 공통 규약만 공유하는 것입니다.

**(a) 유형 A 평가**

현재 구조는 꽤 적절합니다.

```text
src/<pkg>/{datasets,models,experiments}/
docs/{proposal.md, RESULTS.md, exp_logs/LOG.md}
PROJECT.md, LOGGING.md, README.md, pyproject.toml
results/
```

좋은 점은 `python -m <pkg>.experiments.<name>` 실행 방식, editable install, `LOG.md` 기반 실험 인덱싱입니다. ML 파일럿/방법 개발에는 맞습니다.

개선하면 좋은 점:

```text
.
├── README.md
├── PROJECT.md
├── LOGGING.md
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── experiments/
│   │   └── exp001_baseline.yaml
│   └── sweeps/
├── src/
│   └── <pkg>/
│       ├── datasets/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── experiments/
│       ├── utils/
│       └── cli.py
├── scripts/
│   ├── download_data.sh
│   ├── run_exp.sh
│   └── aggregate_results.py
├── notebooks/
│   ├── scratch/
│   └── reports/
├── docs/
│   ├── proposal.md
│   ├── RESULTS.md
│   └── exp_logs/
│       └── LOG.md
├── tests/
├── data/
│   ├── README.md
│   ├── raw/
│   ├── interim/
│   └── processed/
├── artifacts/
│   ├── checkpoints/
│   ├── generated_data/
│   └── runs/
└── results/
    ├── tables/
    ├── figures/
    └── summaries/
```

권장 기준:

- `results/`: 사람이 읽는 최종 결과. 표, 그림, 요약 CSV, 논문용 figure.
- `artifacts/`: 재실행/디버깅용 산출물. checkpoint, tensorboard log, wandb dump, generated dataset, raw predictions.
- `data/`: 원천/전처리 데이터 위치. git에는 보통 내용 제외.
- `configs/`: 실험 재현의 핵심. seed, model, dataset, train params를 전부 config에 둠.
- `notebooks/scratch/`: 탐색용. 중요한 분석은 `.py`나 `notebooks/reports/`로 승격.
- `tests/`: 작더라도 dataset shape, model forward, config load, smoke train 정도는 넣는 게 좋음.

A 유형에서는 `experiments/` 안에 모든 로직을 넣기보다, 실험 파일은 orchestration만 하고 실제 학습/평가 로직은 `training/`, `evaluation/`, `models/`, `datasets/`로 빼는 게 오래 갑니다.

---

**(b) 유형 B 최적 구조**

`data/ models/ xai/ analysis/ paper/`는 나쁘지 않지만, 루트가 약간 산만합니다. XAI/논문형은 **코드, 노트북, 리포트, 논문 산출물**을 분명히 나누는 게 좋습니다.

추천 구조:

```text
.
├── README.md
├── PROJECT.md
├── ROADMAP.md
├── LOGGING.md
├── pyproject.toml
├── configs/
│   ├── dataset.yaml
│   ├── segmentation.yaml
│   ├── xai.yaml
│   └── figures.yaml
├── src/
│   └── <pkg>/
│       ├── data/
│       ├── segmentation/
│       ├── xai/
│       ├── analysis/
│       ├── visualization/
│       └── utils/
├── notebooks/
│   ├── 00_data_inspection.ipynb
│   ├── 01_segmentation_eval.ipynb
│   ├── 02_xai_analysis.ipynb
│   ├── 03_figure_generation.ipynb
│   └── scratch/
├── scripts/
│   ├── prepare_data.py
│   ├── train_segmentation.py
│   ├── run_xai.py
│   ├── export_figures.py
│   └── build_paper_assets.py
├── data/
│   ├── README.md
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── artifacts/
│   ├── checkpoints/
│   ├── masks/
│   ├── xai_maps/
│   └── predictions/
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── diagnostics/
│   └── slides/
├── paper/
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   ├── tables/
│   ├── references.bib
│   └── notes.md
├── docs/
│   ├── method_notes.md
│   ├── dataset_notes.md
│   └── exp_logs/
│       └── LOG.md
└── tests/
```

B 유형의 핵심 규칙:

- `src/<pkg>/xai/`: Grad-CAM, SHAP, attribution, perturbation 등 재사용 가능한 XAI 코드.
- `src/<pkg>/analysis/`: 통계 분석, metric aggregation, correlation, ablation 분석.
- `src/<pkg>/visualization/`: 논문 figure 생성용 함수.
- `notebooks/`: 해석과 검토 중심. 단, 최종 figure는 가능하면 `scripts/export_figures.py`에서 재생성 가능하게.
- `reports/figures/`: 논문에 들어가기 전 검토용 figure.
- `paper/figures/`: 실제 논문에서 참조하는 고정 figure.
- `artifacts/xai_maps/`: heatmap, attribution map, mask prediction 등 대용량 중간 산출물.

즉, 루트에 `xai/ analysis/ models/`를 직접 두기보다 `src/<pkg>/` 아래로 넣는 편이 패키지화와 재현성 면에서 낫습니다.

---

**(c) 공통 규약 체크리스트**

공통으로 유지할 것:

```text
README.md
PROJECT.md
LOGGING.md
docs/exp_logs/LOG.md
configs/
scripts/
src/<pkg>/
data/README.md
artifacts/
results/ or reports/
tests/
.gitignore
pyproject.toml
```

체크리스트:

- `README.md`: 설치, 데이터 준비, 대표 실행 명령, 주요 결과 위치.
- `PROJECT.md`: 연구 질문, 가설, 현재 단계, TODO, 의사결정 기록.
- `LOGGING.md`: 실험 로그 작성법, run id 규칙, artifact 저장 규칙.
- `docs/exp_logs/LOG.md`: 날짜, 실험 ID, 목적, config, seed, 결과, 판정, 산출물 경로.
- `configs/`: 실험 재현 단위. 코드 인자보다 config 우선.
- `scripts/`: 사람이 직접 실행하는 entrypoint.
- `src/<pkg>/`: 재사용 가능한 코드.
- `notebooks/`: 탐색/분석. 중요한 로직은 `src/` 또는 `scripts/`로 이동.
- `tests/`: 최소 smoke test.
- seed는 config에 명시: `seed`, `data_seed`, `model_seed`, `split_seed`.
- 모든 run은 `run_id`를 가짐: 예) `20260706_exp012_baseline_seed42`.
- 결과 저장 시 config snapshot을 같이 저장.
- 가능하면 `artifacts/runs/<run_id>/config.yaml`, `metrics.json`, `stdout.log` 형태 유지.

추천 `LOG.md` 포맷:

```markdown
| Date | Run ID | Purpose | Config | Seed | Result | Decision | Artifacts |
|---|---|---|---|---:|---|---|---|
| 2026-07-06 | exp012_baseline_s42 | baseline check | configs/experiments/exp012.yaml | 42 | acc=0.83 | keep | artifacts/runs/exp012/ |
```

---

**(d) 데이터·큰 파일·체크포인트 관리**

기본 원칙:

- git에는 코드, 설정, 문서, 작은 샘플만.
- 대용량 데이터, 모델 checkpoint, raw output은 git 제외.
- 대신 “어디서 받는지 / 어떤 버전인지 / 어떻게 재생성하는지”를 기록.

추천 `.gitignore`:

```gitignore
# data
data/raw/
data/interim/
data/processed/
data/external/

# artifacts
artifacts/
checkpoints/
wandb/
mlruns/
runs/

# results: 필요하면 선택적으로 추적
results/**/*.pt
results/**/*.pth
results/**/*.ckpt
results/**/*.npy
results/**/*.npz
results/**/*.pkl
results/**/*.parquet

# python
__pycache__/
*.py[cod]
.venv/
.env

# notebooks
.ipynb_checkpoints/
```

관리 방식 선택:

| 상황 | 추천 |
|---|---|
| 혼자 쓰는 소규모 연구 | `.gitignore` + `data/README.md` + 외부 저장 경로 |
| 데이터/모델 버전이 중요함 | DVC |
| checkpoint 몇 개만 공유 | Git LFS |
| KIER 서버/NAS에 데이터 있음 | `data/README.md`에 절대경로/심볼릭 링크 규칙 |
| 논문 재현 패키지 준비 | DVC 또는 release artifact + checksum |

`data/README.md` 예시:

```markdown
# Data

## Raw Data
- Location: /path/to/shared/storage/project_name/raw
- Version: 2026-07-06
- Source: SEM dataset from ...
- Checksum: ...

## Preparation

```bash
python scripts/prepare_data.py --config configs/dataset.yaml
```

## Expected Structure

```text
data/
├── raw/
├── interim/
└── processed/
```
```

체크포인트는 보통 이렇게 둡니다:

```text
artifacts/
└── runs/
    └── 20260706_exp012_baseline_seed42/
        ├── config.yaml
        ├── metrics.json
        ├── model.ckpt
        ├── predictions.parquet
        └── notes.md
```

최종 결론:

- **유형 A**: `src/<pkg>/experiments + configs + artifacts/runs + results` 중심.
- **유형 B**: `src/<pkg>/{data,segmentation,xai,analysis,visualization} + notebooks + reports + paper` 중심.
- 공통 규약은 `README`, `PROJECT`, `LOGGING`, `configs`, `exp_logs`, `data/README`, `.gitignore`, run id, seed, config snapshot만 강하게 통일.
- 구조를 하나로 강제하지 말고, **재현성 단위와 산출물 위치만 통일**하는 게 가장 좋습니다.


OpenAI Codex v0.142.5
--------
workdir: /home/dongwan0629/research/awe
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: none
reasoning summaries: none
session id: 019f3593-b723-7f13-9b1e-e76849037469
--------
user
AI 연구 학생의 여러 프로젝트에 쓸 "연구 레포 구조 템플릿"을 유형별로 최적화하고 싶어. 비판적으로 검토하고 구체적 템플릿을 제안해줘.

## 배경
같은 사람이 성격이 다른 연구 프로젝트를 여럿 함. 구조를 하나로 강제하지 말고 유형별로 최적화하되, 공통 규약은 공유하고 싶음.

## 유형 A — ML 파일럿/방법 개발 (예: AWE, LLM test-time compute)
특징: 실험 스크립트 많음, 모델·데이터생성기·ablation 반복, 빠른 반복, from-scratch 소형모델.
현재 구조(내가 이미 적용):
```
src/<pkg>/{datasets,models,experiments}/   # 코드
docs/{proposal.md, RESULTS.md, exp_logs/LOG.md}
PROJECT.md, LOGGING.md, README.md, pyproject.toml(install -e), results/
```
실행: python -m <pkg>.experiments.<name>. LOG.md는 날짜|실험|컨셉|결과(판정)|산출물 인덱스.

## 유형 B — XAI/분석+논문 (예: PEMFC SEM segmentation+XAI)
특징: 데이터 분석·시각화·XAI 해석 중심, notebook 많음, 최종 산출물이 논문/figure. 다른 에이전트가 만든 구조:
```
data/ models/ xai/ analysis/ paper/  + README.md, ROADMAP.md, .gitignore
```

## 질문
(a) 유형 A 구조는 적절한가? 개선점? (특히 data/ 대용량·notebook·config·seed/재현성·results vs artifacts 분리)
(b) 유형 B(XAI/분석/논문)의 최적 구조는? 위 data/models/xai/analysis/paper가 좋은가, 아니면 더 나은 레이아웃? notebooks/ configs/ reports/ 등 포함 여부.
(c) 두 유형이 공유해야 할 공통 규약(README/PROJECT/LOGGING/exp_logs/.gitignore/데이터 버전관리/재현성)은?
(d) 데이터·큰 파일·모델 체크포인트는 git에서 어떻게(예: .gitignore + DVC/git-lfs/외부저장 포인터)?
각 유형별로 바로 쓸 수 있는 디렉토리 트리 + 공통 규약 체크리스트로. 간결하고 구체적으로.
codex
아래처럼 가는 게 가장 실용적입니다. 핵심은 **A는 “실험 엔진 중심”**, **B는 “분석 산출물/논문 중심”**으로 분리하고, 공통 규약만 공유하는 것입니다.

**(a) 유형 A 평가**

현재 구조는 꽤 적절합니다.

```text
src/<pkg>/{datasets,models,experiments}/
docs/{proposal.md, RESULTS.md, exp_logs/LOG.md}
PROJECT.md, LOGGING.md, README.md, pyproject.toml
results/
```

좋은 점은 `python -m <pkg>.experiments.<name>` 실행 방식, editable install, `LOG.md` 기반 실험 인덱싱입니다. ML 파일럿/방법 개발에는 맞습니다.

개선하면 좋은 점:

```text
.
├── README.md
├── PROJECT.md
├── LOGGING.md
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── experiments/
│   │   └── exp001_baseline.yaml
│   └── sweeps/
├── src/
│   └── <pkg>/
│       ├── datasets/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── experiments/
│       ├── utils/
│       └── cli.py
├── scripts/
│   ├── download_data.sh
│   ├── run_exp.sh
│   └── aggregate_results.py
├── notebooks/
│   ├── scratch/
│   └── reports/
├── docs/
│   ├── proposal.md
│   ├── RESULTS.md
│   └── exp_logs/
│       └── LOG.md
├── tests/
├── data/
│   ├── README.md
│   ├── raw/
│   ├── interim/
│   └── processed/
├── artifacts/
│   ├── checkpoints/
│   ├── generated_data/
│   └── runs/
└── results/
    ├── tables/
    ├── figures/
    └── summaries/
```

권장 기준:

- `results/`: 사람이 읽는 최종 결과. 표, 그림, 요약 CSV, 논문용 figure.
- `artifacts/`: 재실행/디버깅용 산출물. checkpoint, tensorboard log, wandb dump, generated dataset, raw predictions.
- `data/`: 원천/전처리 데이터 위치. git에는 보통 내용 제외.
- `configs/`: 실험 재현의 핵심. seed, model, dataset, train params를 전부 config에 둠.
- `notebooks/scratch/`: 탐색용. 중요한 분석은 `.py`나 `notebooks/reports/`로 승격.
- `tests/`: 작더라도 dataset shape, model forward, config load, smoke train 정도는 넣는 게 좋음.

A 유형에서는 `experiments/` 안에 모든 로직을 넣기보다, 실험 파일은 orchestration만 하고 실제 학습/평가 로직은 `training/`, `evaluation/`, `models/`, `datasets/`로 빼는 게 오래 갑니다.

---

**(b) 유형 B 최적 구조**

`data/ models/ xai/ analysis/ paper/`는 나쁘지 않지만, 루트가 약간 산만합니다. XAI/논문형은 **코드, 노트북, 리포트, 논문 산출물**을 분명히 나누는 게 좋습니다.

추천 구조:

```text
.
├── README.md
├── PROJECT.md
├── ROADMAP.md
├── LOGGING.md
├── pyproject.toml
├── configs/
│   ├── dataset.yaml
│   ├── segmentation.yaml
│   ├── xai.yaml
│   └── figures.yaml
├── src/
│   └── <pkg>/
│       ├── data/
│       ├── segmentation/
│       ├── xai/
│       ├── analysis/
│       ├── visualization/
│       └── utils/
├── notebooks/
│   ├── 00_data_inspection.ipynb
│   ├── 01_segmentation_eval.ipynb
│   ├── 02_xai_analysis.ipynb
│   ├── 03_figure_generation.ipynb
│   └── scratch/
├── scripts/
│   ├── prepare_data.py
│   ├── train_segmentation.py
│   ├── run_xai.py
│   ├── export_figures.py
│   └── build_paper_assets.py
├── data/
│   ├── README.md
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── artifacts/
│   ├── checkpoints/
│   ├── masks/
│   ├── xai_maps/
│   └── predictions/
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── diagnostics/
│   └── slides/
├── paper/
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   ├── tables/
│   ├── references.bib
│   └── notes.md
├── docs/
│   ├── method_notes.md
│   ├── dataset_notes.md
│   └── exp_logs/
│       └── LOG.md
└── tests/
```

B 유형의 핵심 규칙:

- `src/<pkg>/xai/`: Grad-CAM, SHAP, attribution, perturbation 등 재사용 가능한 XAI 코드.
- `src/<pkg>/analysis/`: 통계 분석, metric aggregation, correlation, ablation 분석.
- `src/<pkg>/visualization/`: 논문 figure 생성용 함수.
- `notebooks/`: 해석과 검토 중심. 단, 최종 figure는 가능하면 `scripts/export_figures.py`에서 재생성 가능하게.
- `reports/figures/`: 논문에 들어가기 전 검토용 figure.
- `paper/figures/`: 실제 논문에서 참조하는 고정 figure.
- `artifacts/xai_maps/`: heatmap, attribution map, mask prediction 등 대용량 중간 산출물.

즉, 루트에 `xai/ analysis/ models/`를 직접 두기보다 `src/<pkg>/` 아래로 넣는 편이 패키지화와 재현성 면에서 낫습니다.

---

**(c) 공통 규약 체크리스트**

공통으로 유지할 것:

```text
README.md
PROJECT.md
LOGGING.md
docs/exp_logs/LOG.md
configs/
scripts/
src/<pkg>/
data/README.md
artifacts/
results/ or reports/
tests/
.gitignore
pyproject.toml
```

체크리스트:

- `README.md`: 설치, 데이터 준비, 대표 실행 명령, 주요 결과 위치.
- `PROJECT.md`: 연구 질문, 가설, 현재 단계, TODO, 의사결정 기록.
- `LOGGING.md`: 실험 로그 작성법, run id 규칙, artifact 저장 규칙.
- `docs/exp_logs/LOG.md`: 날짜, 실험 ID, 목적, config, seed, 결과, 판정, 산출물 경로.
- `configs/`: 실험 재현 단위. 코드 인자보다 config 우선.
- `scripts/`: 사람이 직접 실행하는 entrypoint.
- `src/<pkg>/`: 재사용 가능한 코드.
- `notebooks/`: 탐색/분석. 중요한 로직은 `src/` 또는 `scripts/`로 이동.
- `tests/`: 최소 smoke test.
- seed는 config에 명시: `seed`, `data_seed`, `model_seed`, `split_seed`.
- 모든 run은 `run_id`를 가짐: 예) `20260706_exp012_baseline_seed42`.
- 결과 저장 시 config snapshot을 같이 저장.
- 가능하면 `artifacts/runs/<run_id>/config.yaml`, `metrics.json`, `stdout.log` 형태 유지.

추천 `LOG.md` 포맷:

```markdown
| Date | Run ID | Purpose | Config | Seed | Result | Decision | Artifacts |
|---|---|---|---|---:|---|---|---|
| 2026-07-06 | exp012_baseline_s42 | baseline check | configs/experiments/exp012.yaml | 42 | acc=0.83 | keep | artifacts/runs/exp012/ |
```

---

**(d) 데이터·큰 파일·체크포인트 관리**

기본 원칙:

- git에는 코드, 설정, 문서, 작은 샘플만.
- 대용량 데이터, 모델 checkpoint, raw output은 git 제외.
- 대신 “어디서 받는지 / 어떤 버전인지 / 어떻게 재생성하는지”를 기록.

추천 `.gitignore`:

```gitignore
# data
data/raw/
data/interim/
data/processed/
data/external/

# artifacts
artifacts/
checkpoints/
wandb/
mlruns/
runs/

# results: 필요하면 선택적으로 추적
results/**/*.pt
results/**/*.pth
results/**/*.ckpt
results/**/*.npy
results/**/*.npz
results/**/*.pkl
results/**/*.parquet

# python
__pycache__/
*.py[cod]
.venv/
.env

# notebooks
.ipynb_checkpoints/
```

관리 방식 선택:

| 상황 | 추천 |
|---|---|
| 혼자 쓰는 소규모 연구 | `.gitignore` + `data/README.md` + 외부 저장 경로 |
| 데이터/모델 버전이 중요함 | DVC |
| checkpoint 몇 개만 공유 | Git LFS |
| KIER 서버/NAS에 데이터 있음 | `data/README.md`에 절대경로/심볼릭 링크 규칙 |
| 논문 재현 패키지 준비 | DVC 또는 release artifact + checksum |

`data/README.md` 예시:

```markdown
# Data

## Raw Data
- Location: /path/to/shared/storage/project_name/raw
- Version: 2026-07-06
- Source: SEM dataset from ...
- Checksum: ...

## Preparation

```bash
python scripts/prepare_data.py --config configs/dataset.yaml
```

## Expected Structure

```text
data/
├── raw/
├── interim/
└── processed/
```
```

체크포인트는 보통 이렇게 둡니다:

```text
artifacts/
└── runs/
    └── 20260706_exp012_baseline_seed42/
        ├── config.yaml
        ├── metrics.json
        ├── model.ckpt
        ├── predictions.parquet
        └── notes.md
```

최종 결론:

- **유형 A**: `src/<pkg>/experiments + configs + artifacts/runs + results` 중심.
- **유형 B**: `src/<pkg>/{data,segmentation,xai,analysis,visualization} + notebooks + reports + paper` 중심.
- 공통 규약은 `README`, `PROJECT`, `LOGGING`, `configs`, `exp_logs`, `data/README`, `.gitignore`, run id, seed, config snapshot만 강하게 통일.
- 구조를 하나로 강제하지 말고, **재현성 단위와 산출물 위치만 통일**하는 게 가장 좋습니다.
tokens used
5,105

```

## Concise summary

Provider completed successfully. Review the raw output for details.

## Action items

- Review the response and extract decisions you want to apply.
- Capture follow-up implementation tasks if needed.
