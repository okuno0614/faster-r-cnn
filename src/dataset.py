import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import numpy as np

class VacantLandDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Args:
            root_dir (string): 画像が格納されているディレクトリのパス
            annotation_file (string): アノテーションファイルのパス（COCO形式）
            transform (callable, optional): 画像の前処理
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # アノテーションファイルを読み込む
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        # 画像IDとファイル名のマッピングを作成
        self.image_id_to_file = {
            img['id']: img['file_name'] 
            for img in self.annotations['images']
        }
        
        # 画像IDとアノテーションのマッピングを作成
        self.image_id_to_annotations = {}
        for ann in self.annotations['annotations']:
            if ann['image_id'] not in self.image_id_to_annotations:
                self.image_id_to_annotations[ann['image_id']] = []
            self.image_id_to_annotations[ann['image_id']].append(ann)
            
        self.image_ids = list(self.image_id_to_file.keys())
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, self.image_id_to_file[image_id])
        
        # 画像を読み込む
        image = Image.open(image_path).convert('RGB')
        
        # アノテーションを取得
        annotations = self.image_id_to_annotations[image_id]
        
        # バウンディングボックスとラベルを準備
        boxes = []
        labels = []
        
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            # COCO形式から[x1, y1, x2, y2]形式に変換
            boxes.append([
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3]
            ])
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id])
        }
        
        if self.transform:
            image = self.transform(image)
            
        return image, target 