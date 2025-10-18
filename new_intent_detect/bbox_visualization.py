import torch

file_path = "pku_64_1e-06_relu/result/epoch100/design_intent_bbox_train.pt"

bbox_data = torch.load(file_path)
print(f"총 {len(bbox_data)}개 샘플 존재")

# 앞에서 10개 샘플 확인
for i in range(min(30, len(bbox_data))):
    sample = bbox_data[i]
    print(f"\n[{i}]번째 샘플:")
    for key, value in sample.items():
        print(f" - {key}: {value}")
