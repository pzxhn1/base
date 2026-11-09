# 双模态 YOLO11n-OBB 均值融合基线

这是一个可训练、可验证的 **RGB＋热红外双主干**基线。两路各自经过 YOLO11n
主干，在 P3、P4、P5 对应特征上取算术均值，随后使用原版 YOLO11n-OBB
颈部、检测头和损失函数。均值融合没有可学习参数，适合与新的融合模块直接比较。

项目固定使用 `ultralytics==8.4.208`。网络实现见 `dual_yolo/model.py`。

## 1. 安装

```powershell
python -m pip install -r requirements.txt
```

本机已安装依赖，但当前 `torch 2.14.1+cu126` 不支持 RTX 5060 Ti 的 `sm_120`
计算架构；本机验证使用 CPU。正式训练前应安装与 GPU 匹配的 PyTorch 构建。

## 2. 准备数据

本项目需要已经转换成 **Ultralytics OBB 九列格式** 的标签：
`class x1 y1 x2 y2 x3 y3 x4 y4`，坐标归一化到 `[0,1]`。
标签必须在 RGB 参考坐标系下与图像匹配。原始 DroneVehicle 标注可能需要
单独转换；本项目不会猜测原始标注格式或类别编号。

原始目录：

```text
raw/
  train/
    rgb/       001.png ...
    thermal/   001.png ...
    labels/    001.txt ...
  val/
    rgb/
    thermal/
    labels/
```

每对图像必须同名、同尺寸。`prepare_data.py` 检查配对和标签，然后生成
4 通道 TIFF：前三通道为 OpenCV 读取的 BGR，最后一通道为热红外灰度。
训练时模型自动把前三通道调整为 RGB。两种模态共享同一次几何增强。

```powershell
python prepare_data.py --source raw --out packed --names car,truck,freight_car,bus,van
```

`--names` 必须按你实际标签的类别编号填写。生成的 `packed/data.yaml`
含 `channels: 4`。

## 3. 训练与验证

```powershell
python train.py --data packed/data.yaml --model yolo11n-obb.pt --epochs 100 --imgsz 640 --batch 8 --device 0 --project runs/baseline
python evaluate.py --weights runs/baseline/train/weights/best.pt --data packed/data.yaml --imgsz 640 --batch 8 --device 0
```

`.pt` 首次使用时由 Ultralytics 下载预训练权重。可将 `--model` 改为
`yolo11n-obb.yaml` 从随机权重开始。训练器把能匹配的预训练权重载入 RGB
主干及 YOLO 颈部／检测头，然后复制 RGB 主干初始化热红外主干；热红外首层
把三通道卷积权重求和，以适配单通道输入。

输出路径里的 `train` 可能随重复运行变成 `train2` 等，以实际日志为准。

## 4. 可重复的冒烟验证

```powershell
python smoke.py
```

该脚本生成临时配对图像，运行 1 个 CPU 训练轮次和 OBB 验证。
它只证明数据读取、前向、损失、反向、保存和验证流程可运行；合成数据上的
mAP 没有研究意义。本机已运行通过。

## 论文使用边界

- 固定数据划分、类别顺序、标签参考坐标、预训练来源、训练轮次和输入尺寸。
- RGB 单模态、热红外单模态及双主干均值融合应分别训练；本项目实现第三项。
- 模块创新实验应替换 `model.py` 中 P3/P4/P5 的均值操作，保持其余设置一致。
- 真实研究还需昼夜、目标尺寸、跨模态错位分组，以及完整推理延迟。
