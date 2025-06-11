import torch
import torch.nn as nn
import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class VacantLandDetector(nn.Module):
    def __init__(self, num_classes=2):  # 背景と空き地の2クラス
        super(VacantLandDetector, self).__init__()
        # 事前学習済みのFaster R-CNNモデルを読み込む
        self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # 分類器の出力数を調整
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        return self.model(images, targets)
    
    def predict(self, images):
        self.eval()
        with torch.no_grad():
            predictions = self.model(images)
        return predictions 