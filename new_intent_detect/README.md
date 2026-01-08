# New Intent Detect (Design Intent Segmentation)

이 폴더는 포스터/그래픽 이미지에서 텍스트 배치를 위한 디자인 의도 영역(밀집/가용 영역)을 분할(segmentation)하는 모듈입니다. 기본 UNet(mit_b1) 기반이며, 다음 두 가지 모델 타입을 지원합니다.
- design_intent_detector: 이미지(3ch) + saliency_sub(1ch→3ch 반복) 힌트를 멀티스케일 크로스 어텐션으로 융합
- design_intent_detector_simple: 이미지(3ch) 단일 경로(비교군)

## 0) 준비사항
- Python 3.10, CUDA 환경, `requirements.txt` 설치 선행
- 환경변수 설정: 데이터 루트
```bash
export DATASET_ROOT=/abs/path/to/DATA/cgl_pku
```
- 지원 데이터셋: `pku`, `cgl`, `all`
- GPU 지정: 스크립트 내부 `CUDA_VISIBLE_DEVICES` 또는 실행 전 export로 지정

데이터 구조 예(예시는 pku):
```
$DATASET_ROOT/pku/
  ├── annotation/{train.csv,test.csv}
  └── image/
      ├── train/{input,saliency,saliency_sub,closedm,...}
      └── test/{input,saliency,saliency_sub,closedm,...}
```

## 1) 학습: train.sh
스냅샷
```bash
bash train.sh <DATASET> [MODEL_TYPE]
```
- `<DATASET>`: pku | cgl | all
- `[MODEL_TYPE]`(선택):
  - `design_intent_detector`(기본, 이미지+힌트 크로스어텐션)
  - `design_intent_detector_simple`(단일 경로)

train.sh는 내부에서 2회 학습을 실행합니다(예: lr=1e-5, act=relu / lr=1e-3, act=none). 각 러닝은 아래와 같은 실 매개변수로 `main.py`가 호출됩니다.
```bash
python -u main.py \
  --dataset_root $DATASET_ROOT \
  --dataset <DATASET> \
  --batch_size 32 \
  --learning_rate 1e-5 \
  --model_dm_act "relu" \
  --model_type <MODEL_TYPE> \
  --epoch <EPOCHS>
```
- EPOCHS: pku=101, cgl=36, all=26로 자동 설정

예시:
```bash
# pku 데이터셋, 크로스어텐션 모델
bash train.sh pku
# pku 데이터셋, 단일 경로(simple) 모델
bash train.sh pku design_intent_detector_simple
```

출력 디렉토리(예):
```
<EXP_NAME>/
  ├── ckpt/epoch{0..N}.pth
  └── result/epoch{N}/...
```
여기서 `<EXP_NAME>`은 `new_intent_detect/main.py`에서 자동 생성되는 실험명 규칙을 따릅니다
(`{dataset}_{batch}_{lr}_{act}_{model_type}` 형식).

## 2) 추론·피처추출·박스화: test.sh
스냅샷
```bash
bash test.sh <DATASET> <CKPT_PATH> [MODEL_DM_ACT] [BATCH_SIZE]
```
- `<DATASET>`: pku | cgl | all
- `<CKPT_PATH>`: 학습된 체크포인트 경로(예: `.../ckpt/epoch100.pth`)
- `[MODEL_DM_ACT]`(선택): `none|relu|sigmoid` (기본 none)
- `[BATCH_SIZE]`(선택): 기본 32

test.sh는 다음을 순차 수행합니다.
1) 추론(`--infer`) test/train 각각 → 확률맵 저장
2) 피처 추출(`--extract`) test/train 각각 → 7×7 encoder feature 저장
3) `map2box.py`로 확률맵을 박스로 변환

실행 예:
```bash
export DATASET_ROOT=/abs/path/to/DATA/cgl_pku
CKPT=/abs/path/to/pku_28_1e-5_relu_design_intent_detector/ckpt/epoch100.pth
bash test.sh pku $CKPT relu 32
```
출력(예):
```
<EXP_NAME>/result/epoch100/
  ├── test/  # 확률맵(.png) 등 저장
  ├── train/
  ├── eval/  # (run_all.sh 사용 시 평가 결과)
  └── ...
```
피처 저장 경로(추출 시):
```
<EXP_NAME>/result/epoch100/<DATASET>_features/{test,train}/*.npy
```

## 3) 전체 파이프라인: run_all.sh
학습 → 추론·피처 → 평가까지 일괄 수행하며, 다중 러닝레이트/활성화/모델타입 조합을 자동 반복합니다.

스냅샷
```bash
bash run_all.sh \
  [DATASET] [LRs_CSV] [ACTS_CSV] [MODEL_TYPE] \
  [EPOCHS] [BATCH_SIZE] [TEST_INTERVAL] [CHECKPOINT_INTERVAL] \
  [SKIP_TRAINING] [CUDA_DEVICE]
```
기본값:
- DATASET=`pku`
- LRs=`1e-3,1e-4,1e-5,1e-6`
- ACTS=`relu`
- MODEL_TYPE=`design_intent_detector`
- EPOCHS=`101`, BATCH_SIZE=`16`
- TEST_INTERVAL=`20`, CHECKPOINT_INTERVAL=`20`
- SKIP_TRAINING=`false`, CUDA_DEVICE=`6`

예시(기본값 실행):
```bash
export DATASET_ROOT=/abs/path/to/DATA/cgl_pku
bash run_all.sh
```
사용자 지정 예시:
```bash
bash run_all.sh pku "1e-4,1e-5" "relu,sigmoid" design_intent_detector 101 16 20 20 false 0
```
실행 중 요약 정보, 체크포인트·예측·평가 결과 디렉토리를 콘솔에 출력합니다.

## 4) 세그멘테이션 평가 단독: run_eval.sh
이미 생성된 예측 디렉토리와 GT 디렉토리를 이용해 mIoU 등 평가만 수행합니다.

스냅샷
```bash
bash run_eval.sh [pred_dir] [gt_dir] [threshold] [resize_method]
```
기본값:
- pred_dir=`pku_16_0.001_relu/result/epoch100/test`
- gt_dir=`/home/kjk/movers/PosterO-CVPR2025/DATA/cgl_pku/pku/image/test/closedm`
- threshold=`0.5`, resize_method=`bilinear`

예시:
```bash
bash run_eval.sh \
  pku_28_1e-5_relu_design_intent_detector_simple/result/epoch100/test \
  $DATASET_ROOT/pku/image/test/closedm \
  0.5 bilinear
```

## 5) 주요 옵션 설명(핵심)
- `--model_type`: `design_intent_detector`(이미지+saliency_sub 힌트) | `design_intent_detector_simple`(이미지만)
- `--model_dm_act`: UNet 출력 활성화(`none|relu|sigmoid`)
- `--infer / --extract`: 추론/피처 추출 모드 토글
- `--infer_ckpt`: 추론/추출에 사용할 체크포인트 경로
- `--infer_csv`: `train|test` 중 대상 split 지정
- `--epoch`, `--batch_size`, `--learning_rate`: 표준 학습 하이퍼파라미터

## 6) 모델 개요
- design_intent_detector
  - Encoder: mit_b1(이미지), mit_b1(힌트; 1ch→3ch 반복 입력)
  - Multi-scale Cross Attention: encoder 각 stage 채널 맞춰 q(이미지) ← kv(힌트)
  - Decoder/Head: SMP UNet decoder + segmentation head
  - Backward-compatible: 힌트 미제공 시 단일 이미지 경로로 동작
- design_intent_detector_simple
  - 이미지 단일 경로 UNet (비교군)

## 7) 트러블슈팅
- `DATASET_ROOT not set`: `export DATASET_ROOT=/abs/path/to/DATA/cgl_pku` 수행
- 체크포인트 미존재: 학습을 먼저 수행하거나 경로 확인
- CUDA 장치: 스크립트 상단 또는 실행 전 `export CUDA_VISIBLE_DEVICES=<id>`로 지정

---
문의/개선사항은 이 폴더의 스크립트와 `main.py`, `model.py` 주석을 참고하거나 이슈로 남겨주세요.
