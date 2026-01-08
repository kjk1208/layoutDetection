import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from pandas import read_csv
import pandas as pd

class closedm(Dataset):
    def __init__(self, args, preprocess_fn=None, split="train"):
        assert preprocess_fn, "No preprocess functions provided."
        
        #test셋에는 closedm이 없어서 이 if문에서 걸러줌줌
        # 힌트 파일명 접미사 설정 (기본: _mask_pred)
        self.hint_suffix = getattr(args, "saliency_suffix", "_mask_pred")
        # simple 모델일 때는 saliency map을 사용하지 않음
        self.use_saliency = getattr(args, "model_type", "design_intent_detector") != "design_intent_detector_simple"

        if args.dataset == "all":
            if split == "test":
                self.canvas_dir = os.path.join(args.dataset_root, "{}", "image", args.infer_csv, "input")
                self.hint_dir = os.path.join(args.dataset_root, "{}", "image", args.infer_csv, getattr(args, "hint_dir", "saliency_sub"))
                self.closedm_dir = os.path.join(args.dataset_root, "{}", "image", args.infer_csv, "closedm")
                self.df = read_csv(os.path.join(args.dataset_root, "all", "annotation", f"{args.infer_csv}.csv"))
            else:
                self.canvas_dir = os.path.join(args.dataset_root, "{}", "image", split, "input")
                self.hint_dir = os.path.join(args.dataset_root, "{}", "image", split, getattr(args, "hint_dir", "saliency_sub"))
                self.closedm_dir = os.path.join(args.dataset_root, "{}", "image", split, "closedm")
                self.df = read_csv(os.path.join(args.dataset_root, "all", "annotation", f"{split}.csv"))
        else:
            # cgl_pku는 특별한 경우: 내부에 pku와 cgl 서브데이터셋이 있음
            if args.dataset == "cgl_pku":
                if split == "test":
                    self.canvas_dir = os.path.join(args.dataset_root, args.dataset, "{}", "image", args.infer_csv, "input")
                    self.hint_dir = os.path.join(args.dataset_root, args.dataset, "{}", "image", args.infer_csv, getattr(args, "hint_dir", "saliency_sub"))
                    self.closedm_dir = os.path.join(args.dataset_root, args.dataset, "{}", "image", args.infer_csv, "closedm")
                    # pku와 cgl의 annotation 파일을 모두 읽어서 합침
                    dfs = []
                    pku_csv_path = os.path.join(args.dataset_root, args.dataset, "pku", "annotation", f"{args.infer_csv}.csv")
                    if os.path.exists(pku_csv_path):
                        pku_df = read_csv(pku_csv_path)
                        if "dataset" not in pku_df.columns:
                            pku_df["dataset"] = "pku"
                        dfs.append(pku_df)
                    cgl_csv_path = os.path.join(args.dataset_root, args.dataset, "cgl", "annotation", f"{args.infer_csv}.csv")
                    if os.path.exists(cgl_csv_path):
                        cgl_df = read_csv(cgl_csv_path)
                        if "dataset" not in cgl_df.columns:
                            cgl_df["dataset"] = "cgl"
                        dfs.append(cgl_df)
                    if dfs:
                        self.df = pd.concat(dfs, ignore_index=True)
                    else:
                        raise FileNotFoundError(f"Annotation files not found for cgl_pku {args.infer_csv} split")
                else:
                    self.canvas_dir = os.path.join(args.dataset_root, args.dataset, "{}", "image", split, "input")
                    self.hint_dir = os.path.join(args.dataset_root, args.dataset, "{}", "image", split, getattr(args, "hint_dir", "saliency_sub"))
                    self.closedm_dir = os.path.join(args.dataset_root, args.dataset, "{}", "image", split, "closedm")
                    # pku와 cgl의 annotation 파일을 모두 읽어서 합침
                    dfs = []
                    pku_csv_path = os.path.join(args.dataset_root, args.dataset, "pku", "annotation", f"{split}.csv")
                    if os.path.exists(pku_csv_path):
                        pku_df = read_csv(pku_csv_path)
                        if "dataset" not in pku_df.columns:
                            pku_df["dataset"] = "pku"
                        dfs.append(pku_df)
                    cgl_csv_path = os.path.join(args.dataset_root, args.dataset, "cgl", "annotation", f"{split}.csv")
                    if os.path.exists(cgl_csv_path):
                        cgl_df = read_csv(cgl_csv_path)
                        if "dataset" not in cgl_df.columns:
                            cgl_df["dataset"] = "cgl"
                        dfs.append(cgl_df)
                    if dfs:
                        self.df = pd.concat(dfs, ignore_index=True)
                    else:
                        raise FileNotFoundError(f"Annotation files not found for cgl_pku {split} split")
            else:
                if split == "test":
                    self.canvas_dir = os.path.join(args.dataset_root, args.dataset, "image", args.infer_csv, "input")
                    self.hint_dir = os.path.join(args.dataset_root, args.dataset, "image", args.infer_csv, getattr(args, "hint_dir", "saliency_sub"))
                    self.closedm_dir = os.path.join(args.dataset_root, args.dataset, "image", args.infer_csv, "closedm")
                    self.df = read_csv(os.path.join(args.dataset_root, args.dataset, "annotation", f"{args.infer_csv}.csv"))
                else:
                    self.canvas_dir = os.path.join(args.dataset_root, args.dataset, "image", split, "input")
                    self.hint_dir = os.path.join(args.dataset_root, args.dataset, "image", split, getattr(args, "hint_dir", "saliency_sub"))
                    self.closedm_dir = os.path.join(args.dataset_root, args.dataset, "image", split, "closedm")
                    self.df = read_csv(os.path.join(args.dataset_root, args.dataset, "annotation", f"{split}.csv"))

        #정규화
        self.transform_canvas = transforms.Compose([
            lambda x: cv2.resize(x, (224, 224)),
            preprocess_fn,
            transforms.ToTensor()
        ])

        # 힌트(saliency)는 단일 채널로 읽어와 224x224로 리사이즈 후 [0,1] 스케일로 Tensor 변환
        self.transform_hint = transforms.Compose([
            lambda x: cv2.resize(x, (224, 224)),
            transforms.ToTensor()
        ])

        #gt는 정규화 당연히 안해줌
        if split == "train":
            self.transform_closedm = transforms.Compose([
                lambda x: cv2.resize(x, (224, 224)),
                transforms.ToTensor()
            ])

        #annotation csv 파일에서 split을 제외한 나머지 라벨은 지움
        if "split" in self.df:
            if args.extract:
                if args.extract_split == "test":
                    # test 데이터는 test.csv에서 가져오므로 split 필터링 없이 사용
                    pass
                else:
                    self.df = self.df[self.df["split"] == args.extract_split]
            elif not args.infer and split == "train":
                # 학습용 데이터는 train과 valid 모두 사용
                self.df = self.df[self.df["split"].isin(["train", "valid"])]
            elif not args.infer and split == "test":
                # 테스트용 데이터는 test.csv에서 가져옴 (별도 처리)
                pass  # test.csv는 이미 test 데이터만 포함
        #라벨의 중복을 막기 위함
        self.df = self.df.drop_duplicates(subset=['poster_path']).reset_index(drop=True)
        #vis_preview는 데이터셋 확인용
        if split == "test" and args.vis_preview:
            self.df = self.df.iloc[:32]
        
        self.use_all = True if args.dataset == "all" else False
        self.use_cgl_pku = True if args.dataset == "cgl_pku" else False
        self.train = True if split == "train" else False
        
    def __len__(self):
        return len(self.df)
    
    def get_original_images(self, idx):
        """
        transforms 적용 전 원본 이미지를 반환하는 헬퍼 메서드
        테스트 코드에서 사용하기 위해 추가
        """
        entry = self.df.iloc[idx]
        poster_path = entry.poster_path
        entry_dataset = str(entry.dataset) if "dataset" in self.df.columns else str(getattr(self, "dataset", None) or "")
        if entry_dataset == "cgl":
            poster_path = poster_path.replace(".png", ".jpg")
        
        if self.use_all or self.use_cgl_pku:
            ds = str(entry.dataset) if "dataset" in self.df.columns else entry_dataset
            if self.train:
                closedm = cv2.imread(os.path.join(self.closedm_dir.format(ds), poster_path), 0)
            stem, _ = os.path.splitext(poster_path)
            hint_name = f"{stem}{self.hint_suffix}.png"
            hint_path = os.path.join(self.hint_dir.format(ds), hint_name)
            canvas_path = os.path.join(self.canvas_dir.format(ds), poster_path)
        else:
            if self.train:
                closedm = cv2.imread(os.path.join(self.closedm_dir, poster_path), 0)
            stem, _ = os.path.splitext(poster_path)
            hint_name = f"{stem}{self.hint_suffix}.png"
            hint_path = os.path.join(self.hint_dir, hint_name)
            canvas_path = os.path.join(self.canvas_dir, poster_path)
        
        canvas = cv2.imread(canvas_path)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        if self.use_saliency:
            hint = cv2.imread(hint_path, 0)
        else:
            hint = None
        
        if self.train:
            return canvas, hint, closedm
        else:
            return canvas, hint, None
    
    def __getitem__(self, idx):
        #idx에 해당하는 행을 가져옴
        entry = self.df.iloc[idx]
        poster_path = entry.poster_path
        # dataset 컬럼이 없을 수 있음 -> 이 경우 args.dataset 사용
        entry_dataset = str(entry.dataset) if "dataset" in self.df.columns else str(getattr(self, "dataset", None) or "")
        # cgl은 확장자가 jpg인 경우가 있어 보정
        if entry_dataset == "cgl":
            poster_path = poster_path.replace(".png", ".jpg")
        if self.use_all or self.use_cgl_pku:
            #all일때 또는 cgl_pku일때는 dataset 컬럼의 값 가져옴
            ds = str(entry.dataset) if "dataset" in self.df.columns else entry_dataset
            if self.train:
                closedm = cv2.imread(os.path.join(self.closedm_dir.format(ds), poster_path), 0)
            # 힌트 파일명: <stem> + suffix + .png
            stem, _ = os.path.splitext(poster_path)
            hint_name = f"{stem}{self.hint_suffix}.png"
            hint_path = os.path.join(self.hint_dir.format(ds), hint_name)
            canvas_path = os.path.join(self.canvas_dir.format(ds), poster_path)
        else:
            if self.train:
                closedm = cv2.imread(os.path.join(self.closedm_dir, poster_path), 0)
            # 힌트 파일명: <stem> + suffix + .png
            stem, _ = os.path.splitext(poster_path)
            hint_name = f"{stem}{self.hint_suffix}.png"
            hint_path = os.path.join(self.hint_dir, hint_name)
            canvas_path = os.path.join(self.canvas_dir, poster_path)
            
        canvas = cv2.imread(canvas_path)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        if self.use_saliency:
            # 힌트는 그레이스케일로 로드
            hint = cv2.imread(hint_path, 0)
            hint_tensor = self.transform_hint(hint).float()
            # saliency map은 foreground가 선택되어 있지만, 실제로는 선택되지 않은 foreground 영역이 표기 대상이므로 반전
            hint_tensor = 1.0 - hint_tensor
        else:
            # saliency map을 사용하지 않는 경우 더미 힌트 생성
            hint_tensor = torch.zeros(1, 224, 224).float()

        if self.train:
            return self.transform_canvas(canvas).float(), hint_tensor, self.transform_closedm(closedm).float()
        else:
            return self.transform_canvas(canvas).float(), hint_tensor, idx


if __name__ == "__main__":
    """
    테스트 코드: 데이터로더에서 실제 로드하는 데이터들을 샘플 이미지로 저장
    """
    import argparse
    import numpy as np
    from PIL import Image
    
    # 간단한 args 객체 생성
    class Args:
        def __init__(self):
            # 기본값 설정 (필요시 수정)
            # 현재 작업 디렉토리 기준으로 상대 경로 설정
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            
            self.dataset = "cgl_pku"  # 또는 "all", "pku", "cgl" 등
            self.dataset_root = os.path.join(project_root, "DATA")  # 프로젝트 루트의 DATA 폴더
            self.infer_csv = "test"  # test split 사용시
            self.infer = False
            self.extract = False
            self.extract_split = None
            self.vis_preview = False
            self.saliency_suffix = "_mask_pred"
            self.model_type = "design_intent_detector"  # 또는 "design_intent_detector_simple"
            self.hint_dir = "saliency_sub"
    
    args = Args()
    
    # preprocess_fn 정의
    # segmentation_models_pytorch가 설치되어 있으면 실제 preprocess_fn 사용
    try:
        from segmentation_models_pytorch.encoders import get_preprocessing_fn
        preprocess_fn = get_preprocessing_fn('mit_b1', pretrained='imagenet')
        print("✓ segmentation_models_pytorch의 preprocess_fn 사용")
    except ImportError:
        # 설치되어 있지 않으면 간단한 함수 사용
        def simple_preprocess_fn(x):
            # RGB 이미지를 그대로 반환
            return x
        preprocess_fn = simple_preprocess_fn
        print("⚠ segmentation_models_pytorch가 없어 간단한 preprocess_fn 사용")
    
    # 샘플 저장 디렉토리 생성
    output_dir = "test_samples"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "canvas"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "hint"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "closedm"), exist_ok=True)

    # 저장할 샘플 인덱스 지정 (필요에 따라 수정, 예: [0, 3, 5])
    sample_indices = [20,21,22,23,24]
    
    print("=" * 50)
    print("데이터로더 테스트 시작")
    print("=" * 50)
    
    # train split으로 테스트
    try:
        print(f"\n[Train Split 테스트]")
        print(f"Dataset: {args.dataset}")
        print(f"Dataset Root: {args.dataset_root}")
        
        train_dataset = closedm(args=args, preprocess_fn=preprocess_fn, split="train")
        print(f"데이터셋 크기: {len(train_dataset)}")
        
        print(f"\n다음 인덱스 샘플 저장: {sample_indices}")
        
        for i in sample_indices:
            if i >= len(train_dataset):
                print(f"  ✗ 샘플 {i}: 인덱스가 데이터셋 크기({len(train_dataset)})를 초과")
                continue
            try:
                # 원본 이미지 가져오기 (closedm 클래스의 메서드 사용)
                canvas_original, hint_original, closedm_original = train_dataset.get_original_images(i)
                
                # 원본 이미지 저장 (transforms 적용 전)
                Image.fromarray(canvas_original).save(os.path.join(output_dir, "canvas", f"train_{i:03d}_canvas_original.png"))
                
                if hint_original is not None:
                    # hint는 saliency map이므로 반전시켜서 저장 (선택되지 않은 foreground 영역이 표기 대상)
                    hint_original_inverted = 255 - hint_original
                    Image.fromarray(hint_original_inverted, mode='L').save(os.path.join(output_dir, "hint", f"train_{i:03d}_hint_original.png"))
                
                if closedm_original is not None:
                    Image.fromarray(closedm_original, mode='L').save(os.path.join(output_dir, "closedm", f"train_{i:03d}_closedm_original.png"))
                
                # 모델 입력으로 들어가는 transforms가 적용된 tensor 가져오기
                canvas_tensor, hint_tensor, closedm_tensor = train_dataset[i]
                
                # canvas: (C, H, W) -> (H, W, C) - transforms가 적용된 상태
                canvas_np = canvas_tensor.permute(1, 2, 0).numpy()
                # preprocess_fn이 적용되어 정규화되어 있을 수 있으므로, 시각화를 위해 min-max normalization
                # 각 채널별로 [0, 1] 범위로 정규화 후 255로 스케일
                for c in range(canvas_np.shape[2]):
                    channel = canvas_np[:, :, c]
                    if channel.max() > channel.min():
                        canvas_np[:, :, c] = (channel - channel.min()) / (channel.max() - channel.min())
                canvas_np = (canvas_np * 255).astype(np.uint8)
                Image.fromarray(canvas_np).save(os.path.join(output_dir, "canvas", f"train_{i:03d}_canvas.png"))
                
                # hint: (1, H, W) -> (H, W) - transforms가 적용된 상태 (224x224, [0,1] 범위)
                hint_np = hint_tensor.squeeze(0).numpy()
                if hint_np.max() <= 1.0:
                    hint_np = (hint_np * 255).astype(np.uint8)
                else:
                    hint_np = hint_np.astype(np.uint8)
                Image.fromarray(hint_np, mode='L').save(os.path.join(output_dir, "hint", f"train_{i:03d}_hint.png"))
                
                # closedm: (1, H, W) -> (H, W) - transforms가 적용된 상태 (224x224, [0,1] 범위)
                closedm_np = closedm_tensor.squeeze(0).numpy()
                if closedm_np.max() <= 1.0:
                    closedm_np = (closedm_np * 255).astype(np.uint8)
                else:
                    closedm_np = closedm_np.astype(np.uint8)
                Image.fromarray(closedm_np, mode='L').save(os.path.join(output_dir, "closedm", f"train_{i:03d}_closedm.png"))
                
                print(f"  ✓ 샘플 인덱스 {i} 저장 완료")
            except Exception as e:
                print(f"  ✗ 샘플 인덱스 {i} 저장 실패: {e}")
        
        print(f"\nTrain 샘플 저장 완료: {output_dir}/")
        
    except Exception as e:
        print(f"Train split 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # test split으로 테스트
    try:
        print(f"\n[Test Split 테스트]")
        test_dataset = closedm(args=args, preprocess_fn=preprocess_fn, split="test")
        print(f"데이터셋 크기: {len(test_dataset)}")
        
        print(f"\n다음 인덱스 샘플 저장: {sample_indices}")
        
        for i in sample_indices:
            if i >= len(test_dataset):
                print(f"  ✗ 샘플 {i}: 인덱스가 데이터셋 크기({len(test_dataset)})를 초과")
                continue
            try:
                # 원본 이미지 가져오기 (closedm 클래스의 메서드 사용)
                canvas_original, hint_original, _ = test_dataset.get_original_images(i)
                
                # 원본 이미지 저장 (transforms 적용 전)
                Image.fromarray(canvas_original).save(os.path.join(output_dir, "canvas", f"test_{i:03d}_canvas_original.png"))
                
                if hint_original is not None:
                    # hint는 saliency map이므로 반전시켜서 저장 (선택되지 않은 foreground 영역이 표기 대상)
                    hint_original_inverted = 255 - hint_original
                    Image.fromarray(hint_original_inverted, mode='L').save(os.path.join(output_dir, "hint", f"test_{i:03d}_hint_original.png"))
                
                # 모델 입력으로 들어가는 transforms가 적용된 tensor 가져오기
                canvas_tensor, hint_tensor, idx = test_dataset[i]
                
                # canvas: (C, H, W) -> (H, W, C) - transforms가 적용된 상태
                canvas_np = canvas_tensor.permute(1, 2, 0).numpy()
                # preprocess_fn이 적용되어 정규화되어 있을 수 있으므로, 시각화를 위해 min-max normalization
                # 각 채널별로 [0, 1] 범위로 정규화 후 255로 스케일
                for c in range(canvas_np.shape[2]):
                    channel = canvas_np[:, :, c]
                    if channel.max() > channel.min():
                        canvas_np[:, :, c] = (channel - channel.min()) / (channel.max() - channel.min())
                canvas_np = (canvas_np * 255).astype(np.uint8)
                Image.fromarray(canvas_np).save(os.path.join(output_dir, "canvas", f"test_{i:03d}_canvas.png"))
                
                # hint: (1, H, W) -> (H, W) - transforms가 적용된 상태 (224x224, [0,1] 범위)
                hint_np = hint_tensor.squeeze(0).numpy()
                if hint_np.max() <= 1.0:
                    hint_np = (hint_np * 255).astype(np.uint8)
                else:
                    hint_np = hint_np.astype(np.uint8)
                Image.fromarray(hint_np, mode='L').save(os.path.join(output_dir, "hint", f"test_{i:03d}_hint.png"))
                
                print(f"  ✓ 샘플 인덱스 {i} 저장 완료 (idx: {idx})")
            except Exception as e:
                print(f"  ✗ 샘플 인덱스 {i} 저장 실패: {e}")
        
        print(f"\nTest 샘플 저장 완료: {output_dir}/")
        
    except Exception as e:
        print(f"Test split 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("테스트 완료!")
    print(f"샘플 이미지 저장 위치: {os.path.abspath(output_dir)}")
    print("=" * 50)