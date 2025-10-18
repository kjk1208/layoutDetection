import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pandas import read_csv
import argparse
from tqdm import tqdm

def getClosedDM(dataset_root="../../calg_dataset/", annotation_file_name="train_csv_9973.csv", dataset="pku"):
    path = f"{dataset_root}/{dataset}/image/train/input"
    save = f"{dataset_root}/{dataset}/image/train/closedm"
    os.makedirs(save, exist_ok=True)

    # 실제 존재하는 input 이미지 목록
    files = os.listdir(path)
    file_set = set(files)

    # 주석 파일 로딩
    df = read_csv(f"{dataset_root}/{dataset}/annotation/{annotation_file_name}")

    # 확장자 정리: cgl이면 .png → .jpg
    if dataset == "cgl":
        df.poster_path = df.poster_path.str.replace(".png", ".jpg")

    # 실제 이미지가 존재하지 않는 행만 골라냄
    missing_df = df[~df.poster_path.isin(file_set)].reset_index(drop=True)
    kept_df = df[df.poster_path.isin(file_set)].reset_index(drop=True)

    # 삭제된 행 저장
    dropped_csv_path = os.path.join(dataset_root, dataset, "annotation", "dropped_rows.csv")
    kept_csv_path = os.path.join(dataset_root, dataset, "annotation", "cleaned_" + annotation_file_name)
    missing_df.to_csv(dropped_csv_path, index=False)
    kept_df.to_csv(kept_csv_path, index=False)

    print(f"❌ 실제 이미지가 없는 GT 행 수: {len(missing_df)} → {dropped_csv_path}에 저장됨")
    print(f"✅ 남은 주석 데이터 수: {len(kept_df)} → {kept_csv_path}에 저장됨")

    # 이후 전처리 작업에는 cleaned 데이터 사용
    groups = kept_df.groupby("poster_path")
    count = 0

    for f in tqdm(files):
        if f not in groups.groups:
            count += 1
            continue

        img = Image.new("L", (513, 750))
        draw = ImageDraw.Draw(img, "L")
        boxes = groups.get_group(f).box_elem.values
        boxes = [eval(box) for box in boxes]
        for box in boxes:
            if box[0] > box[2]: box[0], box[2] = box[2], box[0]
            if box[1] > box[3]: box[1], box[3] = box[3], box[1]
            draw.rectangle(box, fill="white")

        kernel = np.ones((9, 9), np.uint8)
        img = img.resize((240, 350))
        closed_img = cv2.morphologyEx(np.array(img), cv2.MORPH_CLOSE, kernel)
        Image.fromarray(closed_img).save(os.path.join(save, f))

    print(f"✔ 전처리 완료. GT는 있지만 이미지가 없어서 무시된 이미지 수 (재확인용): {count}")

# def getClosedDM(dataset_root="../../calg_dataset/", annotation_file_name="train_csv_9973.csv", dataset="pku"):
#     path = f"{dataset_root}/{dataset}/image/train/input"
#     save = f"{dataset_root}/{dataset}/image/train/closedm"
#     os.makedirs(save, exist_ok=True)
#     files = os.listdir(path)
#     df = read_csv(f"{dataset_root}/{dataset}/annotation/{annotation_file_name}")
#     if dataset=="cgl":
#         df.poster_path = df.poster_path.str.replace(".png", ".jpg")
#     groups = df.groupby(df.poster_path)
#     count = 0
#     # for f in tqdm(files[:20]):
#     for f in tqdm(files):
#         if f not in groups.groups:
#             count += 1
#             print(f"{count} 주석 없음: {f} → 건너뜀")            
#             continue
#         img = Image.new("L", (513, 750))
#         draw = ImageDraw.Draw(img, "L")
#         query = f
#         boxes = groups.get_group(query).box_elem.values
#         boxes = [eval(box) for box in boxes]
#         for box in boxes:
#             # print(box)
#             if box[0] > box[2]:
#                 box[0], box[2] = box[2], box[0]
#             if box[1] > box[3]:
#                 box[1], box[3] = box[3], box[1]
#             draw.rectangle(box, fill="white")
#         kernel = np.ones((9,9), np.uint8)
#         img = img.resize((240, 350))
#         closed_img = cv2.morphologyEx(np.array(img), cv2.MORPH_CLOSE, kernel)
#         Image.fromarray(closed_img).save(os.path.join(save, f))
#     print(f"주석 없음: {count}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["pku", "cgl"])
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    getClosedDM(dataset_root=args.dataset_root, annotation_file_name="train.csv", dataset=args.dataset)