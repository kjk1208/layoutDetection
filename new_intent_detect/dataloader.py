import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from pandas import read_csv

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
        self.train = True if split == "train" else False
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        #idx에 해당하는 행을 가져옴
        entry = self.df.iloc[idx]
        poster_path = entry.poster_path
        # dataset 컬럼이 없을 수 있음 -> 이 경우 args.dataset 사용
        entry_dataset = str(entry.dataset) if "dataset" in self.df.columns else str(getattr(self, "dataset", None) or "")
        # cgl은 확장자가 jpg인 경우가 있어 보정
        if entry_dataset == "cgl":
            poster_path = poster_path.replace(".png", ".jpg")
        if self.use_all:
            #all일때는 dataset 컬럼의 값 가져옴
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
        else:
            # saliency map을 사용하지 않는 경우 더미 힌트 생성
            hint_tensor = torch.zeros(1, 224, 224).float()

        if self.train:
            return self.transform_canvas(canvas).float(), hint_tensor, self.transform_closedm(closedm).float()
        else:
            return self.transform_canvas(canvas).float(), hint_tensor, idx