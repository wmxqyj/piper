"""
试管架孔位 YOLO 检测器 — 训练脚本

使用流程:
  1. 采集数据

     运行 python run_verification.py（不接机械臂，只开相机），
     按 'c' 保存 RGB 帧到 dataset/rack_holes/images/ 目录。
     采集 80~120 张（不同角度、光照、试管数量）。

  2. 标注数据

     使用 labelImg:
       pip install labelImg
       labelImg src/visual_servo/dataset/rack_holes/images/ \
                --labels tube,empty

     - tube:  孔里有试管，框标注在孔口边缘
     - empty: 空孔，框标注在孔口边缘
     标注结果自动保存到 labels/ 目录（YOLO 格式 .txt）

     每张图约 16 个框，全部框完。至少标注 60 张。

  3. 划分数据集

     按 80/20 随机划分:
       训练集 → images/train/ + labels/train/
       验证集 → images/val/   + labels/val/

  4. 训练

     cd pyAgxArm
     python src/visual_servo/train_rack_detector.py

     训练完成后模型保存在 runs/train/rack_holes/weights/best.pt

  5. 复制模型到 YOLO 配置路径

     cp runs/train/rack_holes/weights/best.pt \
        src/visual_servo/dataset/rack_holes/model.pt

     并在 verification.yaml 中设置:
       yolo:
         model_path: "src/visual_servo/dataset/rack_holes/model.pt"

  6. 运行验证

     python src/visual_servo/verification/run_verification.py

用法:
  python train_rack_detector.py [--model yolov8s.pt] [--epochs 50]
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# 默认路径
_DATASET_DIR = _PROJECT_ROOT / "src" / "visual_servo" / "dataset" / "rack_holes"
_DATASET_YAML = _DATASET_DIR / "dataset.yaml"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "runs" / "train" / "rack_holes"


def check_dataset() -> bool:
    """检查数据集完整性"""
    train_img = _DATASET_DIR / "images" / "train"
    train_lbl = _DATASET_DIR / "labels" / "train"
    val_img = _DATASET_DIR / "images" / "val"
    val_lbl = _DATASET_DIR / "labels" / "val"

    n_train = len(list(train_img.glob("*"))) if train_img.exists() else 0
    n_val = len(list(val_img.glob("*"))) if val_img.exists() else 0

    print(f"数据集: {_DATASET_DIR}")
    print(f"  训练图像: {n_train} 张")
    print(f"  验证图像: {n_val} 张")

    if n_train == 0:
        print("\n[ERROR] 训练集为空！")
        print("请先采集和标注数据:")
        print(f"  1. 采集: 运行 run_verification.py 按 'c' 保存图像到 {_DATASET_DIR}/images/")
        print(f"  2. 标注: labelImg {_DATASET_DIR / 'images'} --labels tube,empty")
        print(f"  3. 划分: 按 80/20 比例复制到 train/ 和 val/")
        return False

    if n_val == 0:
        print("\n[WARN] 验证集为空，将自动从训练集划分 20%")
    return True


def split_dataset(val_ratio: float = 0.2):
    """自动按比例划分训练/验证集"""
    images_dir = _DATASET_DIR / "images"
    labels_dir = _DATASET_DIR / "labels"

    # 如果 images 下有原始文件还没划分
    raw_files = [f for f in images_dir.iterdir() if f.suffix.lower() in
                 (".jpg", ".jpeg", ".png", ".bmp")]

    if not raw_files:
        return

    train_img = images_dir / "train"
    train_lbl = labels_dir / "train"
    val_img = images_dir / "val"
    val_lbl = labels_dir / "val"

    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # 检查是否已经划分过
    if len(list(train_img.iterdir())) > 0:
        print("  数据集已划分，跳过")
        return

    import random
    random.shuffle(raw_files)
    split_idx = int(len(raw_files) * (1 - val_ratio))
    train_files = raw_files[:split_idx]
    val_files = raw_files[split_idx:]

    for f in train_files:
        shutil.copy2(f, train_img / f.name)
        lbl = labels_dir / f"{f.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, train_lbl / lbl.name)

    for f in val_files:
        shutil.copy2(f, val_img / f.name)
        lbl = labels_dir / f"{f.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, val_lbl / lbl.name)

    print(f"  已划分: 训练 {len(train_files)} 张, 验证 {len(val_files)} 张")


def main():
    parser = argparse.ArgumentParser(description="YOLO 试管架孔位检测器训练")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="预训练模型 (yolov8n.pt / yolov8s.pt / yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数 (默认 50)")
    parser.add_argument("--batch", type=int, default=16,
                        help="batch size (默认 16)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸 (默认 640)")
    parser.add_argument("--device", default="0",
                        help="训练设备: 0=GPU, cpu=CPU")
    parser.add_argument("--workers", type=int, default=4,
                        help="数据加载线程数 (默认 4)")
    args = parser.parse_args()

    print("=" * 60)
    print("试管架孔位 YOLO 检测器 — 训练")
    print("=" * 60)
    print(f"模型:       {args.model}")
    print(f"训练轮数:    {args.epochs}")
    print(f"Batch size:  {args.batch}")
    print(f"图像尺寸:    {args.imgsz}")
    print(f"设备:        {args.device}")
    print("=" * 60)

    # 检查数据集
    if not check_dataset():
        sys.exit(1)

    # 自动划分
    print("检查数据集划分...")
    split_dataset()

    # 安装（如果需要）
    try:
        from ultralytics import YOLO
    except ImportError:
        print("正在安装 ultralytics...")
        os.system("pip install ultralytics -q")
        from ultralytics import YOLO

    # 创建模型
    print(f"\n加载预训练模型: {args.model}")
    # 自动下载如果不存在
    model = YOLO(args.model)

    print("开始训练...")
    results = model.train(
        data=str(_DATASET_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(_PROJECT_ROOT / "runs" / "train"),
        name="rack_holes",
        exist_ok=True,
        patience=15,      # early stopping
        augment=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"模型保存: {_DEFAULT_OUTPUT / 'weights' / 'best.pt'}")
    print("=" * 60)
    print("下一步:")
    print(f"  1. 复制模型: cp {_DEFAULT_OUTPUT / 'weights' / 'best.pt'} {_DATASET_DIR / 'model.pt'}")
    print(f"  2. 验证: python src/visual_servo/verification/run_verification.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
