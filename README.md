# 航空衛星画像からの空き地検出

このプロジェクトは、Faster R-CNNを使用して航空衛星画像から空き地を検出するモデルを実装したものです。

## 必要条件

- Python 3.7以上
- PyTorch 1.9.0以上
- torchvision 0.10.0以上
- その他の依存関係は`requirements.txt`を参照

## インストール方法

1. リポジトリをクローン
```bash
git clone [repository-url]
cd [repository-name]
```

2. 仮想環境の作成と有効化
```bash
python -m venv venv
source venv/bin/activate  # Linuxの場合
venv\Scripts\activate     # Windowsの場合
```

3. 依存関係のインストール
```bash
pip install -r requirements.txt
```

## データセットの準備

1. データセットを以下の構造で配置してください：
```
data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── val/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/
    ├── train.json
    └── val.json
```

2. アノテーションファイルはCOCO形式で作成してください。

## 学習方法

以下のコマンドで学習を開始します：
```bash
python src/train.py
```

学習済みモデルは`checkpoints`ディレクトリに保存されます。

## モデルの構造

- バックボーンネットワーク: ResNet-50 with FPN
- 入力: 航空衛星画像
- 出力: 空き地のバウンディングボックスとクラスラベル

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。 