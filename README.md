# YOLO11n-OBB 单模态与双模态基线

本仓库提供两个实验入口。`train_rgb.py` 用原版 Ultralytics YOLO11n-OBB
训练 DroneVehicle 的 **RGB 单模态**；`train.py` 使用下文的 RGB＋热红外均值融合模型。
先完成单模态实验，记录数据和指标口径，再进行双模态比较。

## RGB 单模态：DroneVehicle 全量复现

### 1. 准备 RTX 5060 Ti 环境

在 PowerShell 中进入 `D:\base`。以下命令使用独立的 `.venv`，无需激活环境：

```powershell
cd D:\base
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install 'torch==2.14.1+cu132' 'torchvision==0.29.1+cu132' --extra-index-url https://download.pytorch.org/whl/cu132
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.get_arch_list()); print(torch.ones(1, device='cuda').square().item())"
```

最后一行必须成功输出 `1.0`，并且架构列表包含 `sm_120`。本机已用
PyTorch `2.14.1+cu132` 和 Ultralytics `8.4.208` 验证 GPU 张量运算。
安装版本参见 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/)。
还可运行 `& .\.venv\Scripts\python.exe smoke_rgb.py 0`，用合成图像完整检查
GPU 训练、保存和测试评估；该指标没有研究意义。本机已运行通过。

### 2. 获取官方数据

从 [DroneVehicle 官方页面](https://github.com/VisDrone/DroneVehicle)下载三个压缩包，
原样保存在 `raw\archives`：

| 划分 | 官方链接 | 提取码 |
| --- | --- | --- |
| Train | [train.zip](https://pan.baidu.com/s/1ptZCJ1mKYqFnMnsgqEyoGg) | `ngar` |
| Validation | [val.zip](https://pan.baidu.com/s/1e6e9mESZecpME4IEdU8t3Q) | `jnj6` |
| Test | [test.zip](https://pan.baidu.com/s/1JlXO4jEUQgkR1Vco1hfKhg) | `tqwc` |

解压后应形成以下目录；若压缩包多包了一层目录，请以实际包含 `train/`、
`val/`、`test/` 的那一层作为 `--source`：

```text
raw/DroneVehicle/
  train/trainimg/   train/trainlabel/
  val/valimg/       val/vallabel/
  test/testimg/     test/testlabel/
```

RGB 只用 `*img` 和 `*label`，不把 `*imgr`、`*labelr` 当作 RGB 数据。
原始图像为 840×712，四周各有 100 像素白边；转换脚本会裁为 640×512。
转换包含 RGB XML 中所有有效的五类旋转框和水平框；水平框按轴对齐 OBB 处理。
只有点坐标的对象无法构成检测框，按数量记录并跳过。原版 Ultralytics OBB 不提供
DroneVehicle `difficult` 区域的专门评估规则。请保留 train、val、test 的官方划分，
不重新混合。

### 3. 转换并抽检 OBB 标签

```powershell
& .\.venv\Scripts\python.exe prepare_dronevehicle_rgb.py --source raw\DroneVehicle --out data\dronevehicle_rgb
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

脚本会核对官方每个划分的图像及 XML 数量、配对、尺寸和类别，转换为
Ultralytics 的九列归一化 OBB 标签，输出 `data\dronevehicle_rgb\data.yaml`、
`manifest.json` 和随机抽取的 20 张 `previews`。打开预览图，逐一确认绿色
旋转框落在车辆上，特别检查靠近裁边的车辆。类别编号固定为
`car, freight_car, truck, bus, van`。已核实的 `feright`、`truvk` 笔误分别归入
`freight_car`、`truck`；唯一无法判定的 `*` 类别、点标注和退化框单独计数并跳过。
其他未知类别及图像配对错误会中止转换。`manifest.json` 按划分记录各类标注格式、
保留数量、类别修正与跳过原因。不要带着未解释的异常开始训练。

### 4. GPU 短跑与完整训练

```powershell
& .\.venv\Scripts\python.exe train_rgb.py --epochs 1 --fraction 0.01 --name gpu_probe
& .\.venv\Scripts\python.exe train_rgb.py
```

短跑使用 1% 训练图像，仅验证读取、前向、反向和保存流程，短跑指标不用于论文。
完整训练使用 `yolo11n-obb.pt`、100 轮、`imgsz=640`、`batch=8`、
`seed=42`、`workers=0`。显存不足时用
`--batch 4 --name seed42_batch4` 重新运行，并在论文中记录实际 batch。
训练输出位于 `runs\rgb_obb\seed42`，包括 `args.yaml`、`results.csv`、
曲线和 `weights\best.pt`。若某次运行目录已存在，换一个 `--name`，避免覆盖。
如训练中断，可用 `--resume runs\rgb_obb\seed42\weights\last.pt` 续训；
续训保留检查点中的轮次和训练设置。

### 5. 独立测试

```powershell
& .\.venv\Scripts\python.exe evaluate_rgb.py
```

评估脚本默认读取 `runs\rgb_obb\seed42\weights\best.pt`，仅在官方 test
划分上计算一次指标，并写入 `runs\rgb_obb\seed42\test_metrics.json`。
文件包含总体及每类 Precision、Recall、mAP50、mAP50:95、权重哈希和软件版本。
若训练使用了其他 `--name`，请对应指定 `--weights` 与 `--out`。
训练只使用 train，模型选择只使用 val；合成短跑指标不代表真实检测性能。

## RGB＋热红外均值融合基线

这是一个可训练、可验证的 **RGB＋热红外双主干**基线。两路各自经过 YOLO11n
主干，在 P3、P4、P5 对应特征上取算术均值，随后使用原版 YOLO11n-OBB
颈部、检测头和损失函数。均值融合没有可学习参数，适合与新的融合模块直接比较。

项目固定使用 `ultralytics==8.4.208`。网络实现见 `dual_yolo/model.py`。

### 安装

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

GPU 训练请使用上文的 cu132 独立环境。

### 准备数据

本项目需要已经转换成 **Ultralytics OBB 九列格式** 的标签：
`class x1 y1 x2 y2 x3 y3 x4 y4`，坐标归一化到 `[0,1]`。
标签必须在 RGB 参考坐标系下与图像匹配。原始 DroneVehicle 标注可能需要
单独转换；本项目不会猜测原始标注格式或类别编号。

原始目录：

```text
raw/paired/
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
& .\.venv\Scripts\python.exe prepare_data.py --source raw/paired --out packed --names car,freight_car,truck,bus,van
```

`--names` 必须按你实际标签的类别编号填写。生成的 `packed/data.yaml`
含 `channels: 4`。

### 训练与验证

```powershell
& .\.venv\Scripts\python.exe train.py --data packed/data.yaml --model yolo11n-obb.pt --epochs 100 --imgsz 640 --batch 8 --device 0 --project runs/baseline
& .\.venv\Scripts\python.exe evaluate.py --weights runs/baseline/train/weights/best.pt --data packed/data.yaml --imgsz 640 --batch 8 --device 0
```

`.pt` 首次使用时由 Ultralytics 下载预训练权重。可将 `--model` 改为
`yolo11n-obb.yaml` 从随机权重开始。训练器把能匹配的预训练权重载入 RGB
主干及 YOLO 颈部／检测头，然后复制 RGB 主干初始化热红外主干；热红外首层
把三通道卷积权重求和，以适配单通道输入。

输出路径里的 `train` 可能随重复运行变成 `train2` 等，以实际日志为准。

### 可重复的冒烟验证

```powershell
& .\.venv\Scripts\python.exe smoke.py
```

该脚本生成临时配对图像，运行 1 个 CPU 训练轮次和 OBB 验证。
它只证明数据读取、前向、损失、反向、保存和验证流程可运行；合成数据上的
mAP 没有研究意义。本机已运行通过。

## 论文使用边界

- 固定数据划分、类别顺序、标签参考坐标、预训练来源、训练轮次和输入尺寸。
- RGB 单模态、热红外单模态及双主干均值融合应分别训练；本项目实现 RGB 和双模态均值融合。
- 模块创新实验应替换 `model.py` 中 P3/P4/P5 的均值操作，保持其余设置一致。
- 真实研究还需昼夜、目标尺寸、跨模态错位分组，以及完整推理延迟。
