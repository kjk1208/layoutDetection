#!/usr/bin/env python3
"""
이미지에 segmentation 박스를 그려주는 시각화 스크립트
"""

import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import argparse
import json

def load_bbox_data(pt_file_path):
    """design_intent_bbox_train.pt 파일에서 데이터 로드"""
    try:
        data = torch.load(pt_file_path, weights_only=False)
        print(f"Loaded data type: {type(data)}")
        if isinstance(data, list):
            print(f"Data length: {len(data)}")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"First item keys: {data[0].keys()}")
        return data
    except Exception as e:
        print(f"Error loading {pt_file_path}: {e}")
        return None

def find_image_data(data, image_name):
    """특정 이미지의 데이터 찾기"""
    if not isinstance(data, list):
        return None
    
    for item in data:
        if isinstance(item, dict):
            # 다양한 가능한 키들을 확인
            possible_keys = ['image_path', 'image_name', 'filename', 'file', 'path']
            for key in possible_keys:
                if key in item and image_name in str(item[key]):
                    return item
            
            # 'dataset' 키가 있는 경우 (PosterO 형식)
            if 'dataset' in item and 'image_path' in item:
                if image_name in str(item['image_path']):
                    return item
                    
            # 직접 이미지 이름으로 검색
            for key, value in item.items():
                if isinstance(value, str) and image_name in value:
                    return item
    
    return None

def draw_bbox_on_image(image_path, bbox_data, output_path=None):
    """이미지에 박스 그리기"""
    # 이미지 로드
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # 박스 데이터 추출
    boxes = None
    classes = None
    
    # 다양한 가능한 키들을 확인
    if 'boxes' in bbox_data:
        boxes = bbox_data['boxes']
    elif 'bbox' in bbox_data:
        boxes = bbox_data['bbox']
    elif 'design_intent_bbox' in bbox_data:
        boxes = bbox_data['design_intent_bbox']
    elif 'den_box' in bbox_data:
        boxes = bbox_data['den_box']
    
    if 'classes' in bbox_data:
        classes = bbox_data['classes']
    elif 'cls' in bbox_data:
        classes = bbox_data['cls']
    elif 'labels' in bbox_data:
        classes = bbox_data['labels']
    
    print(f"Found boxes: {boxes}")
    print(f"Found classes: {classes}")
    
    if boxes is None:
        print("No bounding box data found!")
        return image
    
    # 박스 그리기
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()
    if isinstance(classes, torch.Tensor):
        classes = classes.numpy()
    
    # 박스가 리스트인 경우
    if isinstance(boxes, list):
        boxes = np.array(boxes)
    if isinstance(classes, list):
        classes = np.array(classes)
    
    print(f"Boxes shape: {boxes.shape if hasattr(boxes, 'shape') else len(boxes)}")
    print(f"Classes shape: {classes.shape if hasattr(classes, 'shape') else len(classes) if classes is not None else 'None'}")
    
    # 박스 그리기
    for i, box in enumerate(boxes):
        if len(box) >= 4:
            x1, y1, x2, y2 = box[:4]
            # 좌표를 정수로 변환
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 클래스 정보
            cls = classes[i] if classes is not None and i < len(classes) else 1
            
            print(f"Drawing box {i}: ({x1}, {y1}, {x2}, {y2}), class: {cls}")
            
            # 초록색 박스 그리기 (투명도 30%)
            # PIL에서는 투명도를 위해 별도 레이어를 만들어야 함
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # 채워진 박스 (투명도 30%)
            overlay_draw.rectangle([x1, y1, x2, y2], fill=(0, 255, 0, 76))  # 76 = 255 * 0.3
            
            # 테두리 (불투명)
            overlay_draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=2)
            
            # 오버레이를 원본 이미지에 합성
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
    
    # 결과 저장
    if output_path:
        image.save(output_path)
        print(f"Saved visualization to: {output_path}")
    
    return image

def main():
    parser = argparse.ArgumentParser(description="이미지에 segmentation 박스 시각화")
    parser.add_argument("--image_path", required=True, help="이미지 경로")
    parser.add_argument("--pt_file", required=True, help="design_intent_bbox_train.pt 파일 경로")
    parser.add_argument("--output", help="출력 이미지 경로 (선택사항)")
    parser.add_argument("--show", action="store_true", help="이미지 표시")
    
    args = parser.parse_args()
    
    # 데이터 로드
    print("Loading bbox data...")
    data = load_bbox_data(args.pt_file)
    if data is None:
        print("Failed to load bbox data!")
        return
    
    # 이미지 이름 추출
    image_name = os.path.basename(args.image_path)
    print(f"Looking for image: {image_name}")
    
    # 해당 이미지 데이터 찾기
    image_data = find_image_data(data, image_name)
    if image_data is None:
        print(f"No data found for image: {image_name}")
        print("Available data structure:")
        if isinstance(data, list) and len(data) > 0:
            print(json.dumps(data[0], indent=2, default=str))
        return
    
    print(f"Found data for {image_name}:")
    print(json.dumps(image_data, indent=2, default=str))
    
    # 박스 그리기
    print("Drawing bounding boxes...")
    result_image = draw_bbox_on_image(args.image_path, image_data, args.output)
    
    # 이미지 표시
    if args.show:
        result_image.show()

if __name__ == "__main__":
    main()
