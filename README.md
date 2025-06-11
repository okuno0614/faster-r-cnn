# 空き地検出プロジェクト

このプロジェクトは、航空衛星画像から空き地を検出するためのFaster R-CNNモデルを実装したものです。

## プロジェクト構成

```
.
├── src/                    # ソースコード
│   ├── model.py           # Faster R-CNNモデルの定義
│   ├── dataset.py         # データセットクラスの実装
│   ├── train.py           # 学習スクリプト
│   ├── evaluate.py        # 評価スクリプト
│   └── visualization.py   # 可視化モジュール
├── data/                   # データセット
│   ├── images/            # 画像ファイル
│   └── annotations/       # アノテーションファイル
├── notebooks/             # Jupyter Notebooks
│   └── visualization_demo.ipynb  # 可視化デモ
├── checkpoints/           # 学習済みモデルの保存先
├── requirements.txt       # 依存パッケージ
└── README.md             # プロジェクトの説明
```

## 可視化機能

`src/visualization.py`モジュールは、以下の可視化機能を提供します：

1. 基本的な画像表示機能
   - 単一画像の表示
   - バッチ画像の表示
   - バウンディングボックスの可視化

2. データ分析機能
   - クラス分布の可視化
   - バウンディングボックスのサイズ分布
   - 混同行列の表示

3. モデル評価機能
   - 学習曲線の表示
   - PR曲線の表示
   - ROC曲線の表示

4. データ拡張機能
   - 回転、反転、スケーリング
   - エラスティック変換
   - ノイズ追加
   - 明るさ・コントラスト調整

## 可視化デモ

`notebooks/visualization_demo.ipynb`は、可視化モジュールの機能をデモンストレーションするJupyter Notebookです。以下の内容が含まれています：

1. データセットの準備
2. 画像の表示
3. バウンディングボックスの可視化
4. データ拡張の可視化
5. クラス分布の可視化
6. バウンディングボックスのサイズ分布
7. バッチ表示

デモを実行するには：
```bash
jupyter notebook notebooks/visualization_demo.ipynb
```

## インストール方法

1. リポジトリのクローン
```bash
git clone https://github.com/okuno0614/faster-r-cnn.git
cd faster-r-cnn
```

2. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

## データセットの準備

1. データセットを`data/`ディレクトリに配置
   - 画像ファイルを`data/images/`に配置
   - アノテーションファイルを`data/annotations/`に配置

2. アノテーションファイルの形式
   - COCO形式のJSONファイル
   - 訓練データ: `train.json`
   - 検証データ: `val.json`

## 学習方法

1. 学習の実行
```bash
python src/train.py
```

2. 主なパラメータ
   - `--batch-size`: バッチサイズ（デフォルト: 2）
   - `--epochs`: エポック数（デフォルト: 10）
   - `--lr`: 学習率（デフォルト: 0.005）
   - `--momentum`: モメンタム（デフォルト: 0.9）
   - `--weight-decay`: 重み減衰（デフォルト: 0.0005）

## 評価方法

1. 評価の実行
```bash
python src/evaluate.py
```

2. 主なパラメータ
   - `--model-path`: 評価するモデルのパス
   - `--conf-threshold`: 検出の信頼度閾値（デフォルト: 0.5）
   - `--iou-threshold`: IoU閾値（デフォルト: 0.5）

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。 