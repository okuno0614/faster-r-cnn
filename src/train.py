import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import VacantLandDataset
from model import VacantLandDetector
import os
from tqdm import tqdm

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    
    for images, targets in tqdm(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    
    return total_loss / len(data_loader)

def main():
    # ハイパーパラメータ
    num_epochs = 10
    batch_size = 4
    learning_rate = 0.005
    
    # パスの設定
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    images_dir = os.path.join(base_dir, 'data', 'images')
    train_ann_file = os.path.join(base_dir, 'data', 'annotations', 'train.json')
    val_ann_file = os.path.join(base_dir, 'data', 'annotations', 'val.json')
    checkpoints_dir = os.path.join(base_dir, 'checkpoints')
    
    # チェックポイントディレクトリの作成
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # データの前処理
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # データセットの準備
    train_dataset = VacantLandDataset(
        root_dir=images_dir,
        annotation_file=train_ann_file,
        transform=transform
    )
    
    val_dataset = VacantLandDataset(
        root_dir=images_dir,
        annotation_file=val_ann_file,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # モデルの初期化
    model = VacantLandDetector()
    model.to(device)
    
    # オプティマイザの設定
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # 学習ループ
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # 学習
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f'Training Loss: {train_loss:.4f}')
        
        # 検証
        val_loss = evaluate(model, val_loader, device)
        print(f'Validation Loss: {val_loss:.4f}')
        
        # モデルの保存
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)

if __name__ == '__main__':
    main() 