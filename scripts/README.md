# scripts 디렉터리 개요

데이터 정합/검증/분할/이동 유틸리티 스크립트 모음입니다. 경로는 예시이므로 환경에 맞게 조정하세요.

## compare_train_files.py
- 역할: image/train 하위의 input, original, closedm, saliency_sub 4개 폴더를 파일 키(stem) 기준으로 비교하고 누락 파일을 리포트(txt)로 저장
- 살리언시 규칙: 키 + _mask_pred + .png 로 매칭
- 출력: train_folder_mismatches.txt (base_dir 하위)
- 예시
  ```bash
  python compare_train_files.py \
    --base_dir /home/kjk/movers/PosterO-CVPR2025/RALF/DATA/pku/image/train \
    --saliency saliency_sub --saliency_suffix _mask_pred
  ```

## check_train_csv.py
- 역할: annotation/train.csv 의 poster_path 목록과 실제 파일 존재 여부를 비교하여 누락 수를 콘솔에 출력(파일 수정 없음)
- 살리언시 규칙: 키 + _mask_pred + .png 로 매칭
- 예시
  ```bash
  python check_train_csv.py \
    --dataset_root /home/kjk/movers/PosterO-CVPR2025/RALF/DATA \
    --dataset pku \
    --csv train.csv \
    --image_split train \
    --saliency_dir saliency_sub --saliency_suffix _mask_pred
  ```

## clean_train_csv.py
- 역할: 실제 파일이 모두 존재하는 행만 남겨 cleaned_train.csv 생성, 누락 행은 dropped_rows.csv 저장
- 대상: image/<split>/{input, original, closedm, saliency_sub}
- 예시
  ```bash
  python clean_train_csv.py \
    --dataset_root /home/kjk/movers/PosterO-CVPR2025/RALF/DATA \
    --dataset pku \
    --csv train.csv \
    --image_split train \
    --saliency_dir saliency_sub --saliency_suffix _mask_pred
  ```

## split_csv_by_poster.py
- 역할: poster_path 단위 그룹을 보존하여 지정 비율(기본 9:1)로 train/test CSV 분할
- 출력: train_split.csv, test_split.csv (입력 CSV와 동일 폴더)
- 예시
  ```bash
  python split_csv_by_poster.py \
    --input_csv /home/kjk/movers/PosterO-CVPR2025/DATA/cgl_pku/pku/annotation/train.csv \
    --ratio 9:1 --seed 42
  ```

## move_test_files.py
- 역할: annotation/test.csv 기준으로 image/train/{input, original, closedm, saliency_sub} 파일을 image/test/{...}로 이동
- 확장자 처리: .png 우선, 없으면 .jpg 검사; 살리언시는 접미사(_mask_pred) 적용
- 드라이런: --dry_run 으로 이동 계획만 출력
- 예시
  ```bash
  # 계획만 출력
  python move_test_files.py \
    --dataset_root /home/kjk/movers/PosterO-CVPR2025/DATA/cgl_pku \
    --dataset pku \
    --csv /home/kjk/movers/PosterO-CVPR2025/DATA/cgl_pku/pku/annotation/test.csv \
    --dry_run

  # 실제 이동
  python move_test_files.py \
    --dataset_root /home/kjk/movers/PosterO-CVPR2025/DATA/cgl_pku \
    --dataset pku \
    --csv /home/kjk/movers/PosterO-CVPR2025/DATA/cgl_pku/pku/annotation/test.csv
  ```

---
- 공통 팁: 살리언시 폴더/접미사가 다르면 --saliency_dir, --saliency_suffix 로 조정
