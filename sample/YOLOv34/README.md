[简体中文](./README.md)

# YOLOv34

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
  * [6.1 测试方法](#61-测试方法)
  * [6.2 测试结果](#62-测试结果)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
* [8.FAQ](#8-faq)
  
## 1. 简介

作为一种经典的单阶段目标检测框架，YOLO系列的目标检测算法得到了学术界与工业界们的广泛关注。由于YOLO系列属于单阶段目标检测，因而具有较快的推理速度，能够更好的满足现实场景的需求。随着YOLOv3算法的出现，使得YOLO系列的检测算达到了高潮。YOLOv4则是在YOLOv3算法的基础上增加了很多实用的技巧，使得它的速度与精度都得到了极大的提升。本例程对YOLOv4和YOLOv3模型和算法进行移植，使之能在SOPHON BM1684\BM1684X\BM1688上进行推理测试。


**参考repo:** [​YOLOv3 Pytorch开源仓库](https://github.com/ultralytics/yolov3)[​YOLOv4 Pytorch开源仓库](https://github.com/bubbliiiing/yolov4-pytorch)


## 2. 特性
* 支持BM1688(SoC)、BM1684X(x86 PCIe、SoC、riscv PCIe)、BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688)、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持单batch和多batch模型推理
* 支持1个输出和3个输出模型推理
* 支持图片和视频测试
 
## 3. 准备模型与数据
本例程采用的YOLOv3模型权重通过命令`model = torch.hub.load('ultralytics/yolov3', 'yolov3', pretrained=True)`进行下载，YOLOv4模型权重来自于[​YOLOv4_weights.pth](https://github.com/bubbliiiing/yolov4-pytorch/releases/download/v1.0/yolo4_weights.pth)。

建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。具体可参考[YOLOv34模型导出](./docs/YOLOv34_Export_Guide.md)。

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
├── BM1684
│   ├── yolov3_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov3_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   ├── yolov3_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
│   ├── yolov4_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   ├── yolov4_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=1
│   └── yolov4_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684的INT8 BModel，batch_size=4
├── BM1684X
│   ├── yolov3_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov3_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov3_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   ├── yolov3_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
│   ├── yolov4_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov4_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   ├── yolov4_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
│   └── yolov4_int8_4b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=4
├── BM1688
│   ├── yolov3_fp32_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1
│   ├── yolov3_fp16_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1
│   ├── yolov3_int8_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1
│   ├── yolov3_int8_4b.bmodel       # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4
│   ├── yolov4_fp32_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1
│   ├── yolov4_fp16_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1
│   ├── yolov4_int8_1b.bmodel       # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1
│   ├── yolov4_int8_4b.bmodel       # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4
│   ├── yolov3_fp32_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── yolov3_fp16_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   ├── yolov3_int8_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   ├── yolov3_int8_4b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=2
│   ├── yolov4_fp32_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── yolov4_fp16_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
│   ├── yolov4_int8_1b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=1, num_core=2
│   └── yolov4_int8_4b_2core.bmodel # 使用TPU-MLIR编译，用于BM1688的INT8 BModel，batch_size=4, num_core=2
└── onnx
    ├── yolov3.onnx             # 导出的yolov3 onnx动态模型       
    ├── yolov4_1b.onnx          # 导出的yolov4 1batch onnx模型  
    └── yolov4_4b.onnx          # 导出的yolov4 4batch onnx模型  
```
下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── test_car_person_1080P.mp4                 # 测试视频
├── coco.names                                # coco类别名文件
├── coco128                                   # coco128数据集，用于模型量化
└── coco                                      
    ├── val2017_1000                          # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
    └── instances_val2017_1000.json           # coco val2017_1000数据集标签文件，用于计算精度评价指标  
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`yolov3_fp32bmodel_mlir.sh`和`yolov4_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688**），如：

```bash
./scripts/yolov3_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

​执行上述命令会在`models/BM1684`下生成`yolov3_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

```bash
./scripts/yolov4_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

​执行上述命令会在`models/BM1684`下生成`yolov4_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`yolov3_fp16bmodel_mlir.sh`和`yolov4_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688**），如：

```bash
./scripts/yolov3_fp16bmodel_mlir.sh bm1684x #bm1688
```

​执行上述命令会在`models/BM1684X/`下生成`yolov3_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

```bash
./scripts/yolov4_fp16bmodel_mlir.sh bm1684x #bm1688
```

​执行上述命令会在`models/BM1684X/`下生成`yolov4_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`yolov3_int8bmodel_mlir.sh`和`yolov4_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684/BM1684X**），如：

```shell
./scripts/yolov3_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

​上述脚本会在`models/BM1684`下生成`yolov3_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

```shell
./scripts/yolov4_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

​上述脚本会在`models/BM1684`下生成`yolov4_int8_1b.bmodel`等文件，即转换好的INT8 BModel。建议在转换1684 int8 4b模型时,在脚本中model_deploy.py参数中添加qtable，即--quantize_table ../models/yolov4_4b_int8_qtable \。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov3_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 测试结果
在coco2017val_1000数据集上，**推理时设置参数：--conf_thresh=0.001 --nms_thresh=0.6**，yolov3精度测试结果如下：
|   测试平台    |      测试程序     |              测试模型               |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ----------------------------------- | ------------- | -------- |
| BM1684 PCIe  | yolov34_opencv.py | yolov3_fp32_1b.bmodel | 0.471         | 0.663    |
| BM1684 PCIe  | yolov34_opencv.py | yolov3_int8_1b.bmodel | 0.383         | 0.623    |
| BM1684 PCIe  | yolov34_bmcv.py   | yolov3_fp32_1b.bmodel | 0.458         | 0.657    |
| BM1684 PCIe  | yolov34_bmcv.py   | yolov3_int8_1b.bmodel | 0.371         | 0.614    |
| BM1684 PCIe  | yolov34_bmcv.pcie | yolov3_fp32_1b.bmodel | 0.446         | 0.650    |
| BM1684 PCIe  | yolov34_bmcv.pcie | yolov3_int8_1b.bmodel | 0.370         | 0.614    |
| BM1684 PCIe  | yolov34_sail.pcie | yolov3_fp32_1b.bmodel | 0.446         | 0.650    |
| BM1684 PCIe  | yolov34_sail.pcie | yolov3_int8_1b.bmodel | 0.370         | 0.614    |
| BM1684X PCIe | yolov34_opencv.py | yolov3_fp32_1b.bmodel | 0.470         | 0.663    |
| BM1684X PCIe | yolov34_opencv.py | yolov3_fp16_1b.bmodel | 0.470         | 0.663    |
| BM1684X PCIe | yolov34_opencv.py | yolov3_int8_1b.bmodel | 0.446         | 0.657    |
| BM1684X PCIe | yolov34_bmcv.py   | yolov3_fp32_1b.bmodel | 0.458         | 0.657    |
| BM1684X PCIe | yolov34_bmcv.py   | yolov3_fp16_1b.bmodel | 0.458         | 0.657    |
| BM1684X PCIe | yolov34_bmcv.py   | yolov3_int8_1b.bmodel | 0.436         | 0.651    |
| BM1684X PCIe | yolov34_bmcv.pcie | yolov3_fp32_1b.bmodel | 0.446         | 0.649    |
| BM1684X PCIe | yolov34_bmcv.pcie | yolov3_fp16_1b.bmodel | 0.446         | 0.649    |
| BM1684X PCIe | yolov34_bmcv.pcie | yolov3_int8_1b.bmodel | 0.427         | 0.641    |
| BM1684X PCIe | yolov34_sail.pcie | yolov3_fp32_1b.bmodel | 0.446         | 0.649    |
| BM1684X PCIe | yolov34_sail.pcie | yolov3_fp16_1b.bmodel | 0.446         | 0.649    |
| BM1684X PCIe | yolov34_sail.pcie | yolov3_int8_1b.bmodel | 0.427         | 0.641    |
| BM1688 SoC   | yolov34_opencv.py | yolov3_fp32_1b.bmodel | 0.471         | 0.663    |
| BM1688 SoC   | yolov34_opencv.py | yolov3_fp16_1b.bmodel | 0.471         | 0.663    |
| BM1688 SoC   | yolov34_opencv.py | yolov3_int8_1b.bmodel | 0.445         | 0.656    |
| BM1688 SoC   | yolov34_bmcv.py   | yolov3_fp32_1b.bmodel | 0.458         | 0.657    |
| BM1688 SoC   | yolov34_bmcv.py   | yolov3_fp16_1b.bmodel | 0.458         | 0.657    |
| BM1688 SoC   | yolov34_bmcv.py   | yolov3_int8_1b.bmodel | 0.435         | 0.650    |
| BM1688 SoC   | yolov34_bmcv.soc  | yolov3_fp32_1b.bmodel | 0.446         | 0.649    |
| BM1688 SoC   | yolov34_bmcv.soc  | yolov3_fp16_1b.bmodel | 0.446         | 0.649    |
| BM1688 SoC   | yolov34_bmcv.soc  | yolov3_int8_1b.bmodel | 0.427         | 0.641    |
| BM1688 SoC   | yolov34_sail.soc  | yolov3_fp32_1b.bmodel | 0.446         | 0.649    |
| BM1688 SoC   | yolov34_sail.soc  | yolov3_fp16_1b.bmodel | 0.446         | 0.649    |
| BM1688 SoC   | yolov34_sail.soc  | yolov3_int8_1b.bmodel | 0.427         | 0.641    |
|   SRM1-20    | yolov34_opencv.py | yolov3_fp32_1b.bmodel | 0.471         | 0.663    |
|   SRM1-20    | yolov34_opencv.py | yolov3_fp16_1b.bmodel | 0.471         | 0.663    |
|   SRM1-20    | yolov34_opencv.py | yolov3_int8_1b.bmodel | 0.445         | 0.656    |
|   SRM1-20    | yolov34_opencv.py | yolov3_int8_4b.bmodel | 0.444         | 0.655    |
|   SRM1-20    | yolov34_bmcv.py   | yolov3_fp32_1b.bmodel | 0.458         | 0.657    |
|   SRM1-20    | yolov34_bmcv.py   | yolov3_fp16_1b.bmodel | 0.458         | 0.657    |
|   SRM1-20    | yolov34_bmcv.py   | yolov3_int8_1b.bmodel | 0.436         | 0.651    |
|   SRM1-20    | yolov34_bmcv.py   | yolov3_int8_4b.bmodel | 0.435         | 0.650    |
|   SRM1-20    | yolov34_bmcv.pcie | yolov3_fp32_1b.bmodel | 0.446         | 0.649    |
|   SRM1-20    | yolov34_bmcv.pcie | yolov3_fp16_1b.bmodel | 0.446         | 0.649    |
|   SRM1-20    | yolov34_bmcv.pcie | yolov3_int8_1b.bmodel | 0.427         | 0.641    |
|   SRM1-20    | yolov34_bmcv.pcie | yolov3_int8_4b.bmodel | 0.427         | 0.641    |
|   SRM1-20    | yolov34_sail.pcie | yolov3_fp32_1b.bmodel | 0.446         | 0.649    |
|   SRM1-20    | yolov34_sail.pcie | yolov3_fp16_1b.bmodel | 0.446         | 0.649    |
|   SRM1-20    | yolov34_sail.pcie | yolov3_int8_1b.bmodel | 0.427         | 0.641    |
|   SRM1-20    | yolov34_sail.pcie | yolov3_int8_4b.bmodel | 0.427         | 0.641    |

在coco2017val_1000数据集上，**推理时设置参数：--conf_thresh=0.3 --nms_thresh=0.5**，yolov4精度测试结果如下：
|   测试平台    |      测试程序     |              测试模型               |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ----------------------------------- | ------------- | -------- |
| BM1684 PCIe  | yolov34_opencv.py | yolov4_fp32_1b.bmodel | 0.252         | 0.526    |
| BM1684 PCIe  | yolov34_opencv.py | yolov4_int8_1b.bmodel | 0.208         | 0.460    |
| BM1684 PCIe  | yolov34_bmcv.py   | yolov4_fp32_1b.bmodel | 0.242         | 0.504    |
| BM1684 PCIe  | yolov34_bmcv.py   | yolov4_int8_1b.bmodel | 0.200         | 0.436    |
| BM1684 PCIe  | yolov34_bmcv.pcie | yolov4_fp32_1b.bmodel | 0.248         | 0.524    |
| BM1684 PCIe  | yolov34_bmcv.pcie | yolov4_int8_1b.bmodel | 0.204         | 0.457    |
| BM1684 PCIe  | yolov34_sail.pcie | yolov4_fp32_1b.bmodel | 0.246         | 0.521    |
| BM1684 PCIe  | yolov34_sail.pcie | yolov4_int8_1b.bmodel | 0.204         | 0.457    |
| BM1684X PCIe | yolov34_opencv.py | yolov4_fp32_1b.bmodel | 0.252         | 0.527    |
| BM1684X PCIe | yolov34_opencv.py | yolov4_fp16_1b.bmodel | 0.252         | 0.526    |
| BM1684X PCIe | yolov34_opencv.py | yolov4_int8_1b.bmodel | 0.237         | 0.499    |
| BM1684X PCIe | yolov34_bmcv.py   | yolov4_fp32_1b.bmodel | 0.244         | 0.509    |
| BM1684X PCIe | yolov34_bmcv.py   | yolov4_fp16_1b.bmodel | 0.240         | 0.501    |
| BM1684X PCIe | yolov34_bmcv.py   | yolov4_int8_1b.bmodel | 0.225         | 0.472    |
| BM1684X PCIe | yolov34_bmcv.pcie | yolov4_fp32_1b.bmodel | 0.247         | 0.523    |
| BM1684X PCIe | yolov34_bmcv.pcie | yolov4_fp16_1b.bmodel | 0.247         | 0.524    |
| BM1684X PCIe | yolov34_bmcv.pcie | yolov4_int8_1b.bmodel | 0.232         | 0.496    |
| BM1684X PCIe | yolov34_sail.pcie | yolov4_fp32_1b.bmodel | 0.247         | 0.523    |
| BM1684X PCIe | yolov34_sail.pcie | yolov4_fp16_1b.bmodel | 0.247         | 0.524    |
| BM1684X PCIe | yolov34_sail.pcie | yolov4_int8_1b.bmodel | 0.232         | 0.496    |
| BM1688 SoC   | yolov34_opencv.py | yolov4_fp32_1b.bmodel | 0.252         | 0.526    |
| BM1688 SoC   | yolov34_opencv.py | yolov4_fp16_1b.bmodel | 0.252         | 0.526    |
| BM1688 SoC   | yolov34_opencv.py | yolov4_int8_1b.bmodel | 0.234         | 0.495    |
| BM1688 SoC   | yolov34_bmcv.py   | yolov4_fp32_1b.bmodel | 0.244         | 0.509    |
| BM1688 SoC   | yolov34_bmcv.py   | yolov4_fp16_1b.bmodel | 0.244         | 0.509    |
| BM1688 SoC   | yolov34_bmcv.py   | yolov4_int8_1b.bmodel | 0.226         | 0.475    |
| BM1688 SoC   | yolov34_bmcv.soc  | yolov4_fp32_1b.bmodel | 0.247         | 0.521    |
| BM1688 SoC   | yolov34_bmcv.soc  | yolov4_fp16_1b.bmodel | 0.247         | 0.521    |
| BM1688 SoC   | yolov34_bmcv.soc  | yolov4_int8_1b.bmodel | 0.229         | 0.491    |
| BM1688 SoC   | yolov34_sail.soc  | yolov4_fp32_1b.bmodel | 0.246         | 0.520    |
| BM1688 SoC   | yolov34_sail.soc  | yolov4_fp16_1b.bmodel | 0.247         | 0.521    |
| BM1688 SoC   | yolov34_sail.soc  | yolov4_int8_1b.bmodel | 0.229         | 0.491    |
|   SRM1-20    | yolov34_opencv.py | yolov4_fp32_1b.bmodel | 0.381         | 0.557    |
|   SRM1-20    | yolov34_opencv.py | yolov4_fp16_1b.bmodel | 0.381         | 0.557    |
|   SRM1-20    | yolov34_opencv.py | yolov4_int8_1b.bmodel | 0.320         | 0.528    |
|   SRM1-20    | yolov34_opencv.py | yolov4_int8_4b.bmodel | 0.319         | 0.527    |
|   SRM1-20    | yolov34_bmcv.py   | yolov4_fp32_1b.bmodel | 0.368         | 0.544    |
|   SRM1-20    | yolov34_bmcv.py   | yolov4_fp16_1b.bmodel | 0.368         | 0.544    |
|   SRM1-20    | yolov34_bmcv.py   | yolov4_int8_1b.bmodel | 0.304         | 0.508    |
|   SRM1-20    | yolov34_bmcv.py   | yolov4_int8_4b.bmodel | 0.305         | 0.512    |
|   SRM1-20    | yolov34_bmcv.pcie | yolov4_fp32_1b.bmodel | 0.376         | 0.553    |
|   SRM1-20    | yolov34_bmcv.pcie | yolov4_fp16_1b.bmodel | 0.375         | 0.552    |
|   SRM1-20    | yolov34_bmcv.pcie | yolov4_int8_1b.bmodel | 0.310         | 0.522    |
|   SRM1-20    | yolov34_bmcv.pcie | yolov4_int8_4b.bmodel | 0.308         | 0.522    |
|   SRM1-20    | yolov34_sail.pcie | yolov4_fp32_1b.bmodel | 0.376         | 0.553    |
|   SRM1-20    | yolov34_sail.pcie | yolov4_fp16_1b.bmodel | 0.375         | 0.552    |
|   SRM1-20    | yolov34_sail.pcie | yolov4_int8_1b.bmodel | 0.310         | 0.522    |
|   SRM1-20    | yolov34_sail.pcie | yolov4_int8_4b.bmodel | 0.308         | 0.522    |

> **测试说明**：  
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2. BM1688 1core和BM1688 2core的模型精度基本一致；
> 3. 由于sdk版本之间可能存在差异，实际运行结果与本表有<1%的精度误差是正常的；

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/yolov3_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                   | calculate time(ms) |
| ------------------------------------------- | ----------------- |
| BM1684/yolov3_fp32_1b.bmodel        | 102.2              |
| BM1684/yolov3_int8_1b.bmodel        | 53.7               |
| BM1684/yolov3_int8_4b.bmodel        | 19.3               |
| BM1684X/yolov3_fp32_1b.bmodel       | 156.5              |
| BM1684X/yolov3_fp16_1b.bmodel       | 22.2               |
| BM1684X/yolov3_int8_1b.bmodel       | 9.3                |
| BM1684X/yolov3_int8_4b.bmodel       | 9.2                |
| BM1688/yolov3_fp32_1b.bmodel        | 773.9              |
| BM1688/yolov3_fp16_1b.bmodel        | 136.9              |
| BM1688/yolov3_int8_1b.bmodel        | 32.9               |
| BM1688/yolov3_int8_4b.bmodel        | 29.7               |
| BM1688/yolov3_fp32_1b_2core.bmodel  | 694.2              |
| BM1688/yolov3_fp16_1b_2core.bmodel  | 89.9               |
| BM1688/yolov3_int8_1b_2core.bmodel  | 25.8               |
| BM1688/yolov3_int8_4b_2core.bmodel  | 17.8               |
| BM1684/yolov4_fp32_1b.bmodel        | 77.5               |
| BM1684/yolov4_int8_1b.bmodel        | 28.6               |
| BM1684/yolov4_int8_4b.bmodel        | 13.4               |
| BM1684X/yolov4_fp32_1b.bmodel       | 70.4               |
| BM1684X/yolov4_fp16_1b.bmodel       | 13.6               |
| BM1684X/yolov4_int8_1b.bmodel       | 6.1                |
| BM1684X/yolov4_int8_4b.bmodel       | 5.6                |
| BM1688/yolov4_fp32_1b.bmodel        | 375.3              |
| BM1688/yolov4_fp16_1b.bmodel        | 94.0               |
| BM1688/yolov4_int8_1b.bmodel        | 18.1               |
| BM1688/yolov4_int8_4b.bmodel        | 15.5               |
| BM1688/yolov4_fp32_1b_2core.bmodel  | 277.5              |
| BM1688/yolov4_fp16_1b_2core.bmodel  | 63.8               |
| BM1688/yolov4_int8_1b_2core.bmodel  | 15.5               |
| BM1688/yolov4_int8_4b_2core.bmodel  | 9.4                |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.001，nms_thresh=0.6，yolov3性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型       |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ----------------- | --------------------- | -------- | ---------     | ---------     | ---------- |
| BM1684 SoC  | yolov34_opencv.py | yolov3_fp32_1b.bmodel | 15.3     | 25.1          | 112.0         | 159.2      |
| BM1684 SoC  | yolov34_opencv.py | yolov3_int8_1b.bmodel | 15.1     | 24.7          | 63.5          | 157.6      |
| BM1684 SoC  | yolov34_opencv.py | yolov3_int8_4b.bmodel | 15.0     | 23.2          | 27.5          | 152.7      |
| BM1684 SoC  | yolov34_bmcv.py   | yolov3_fp32_1b.bmodel | 3.6      | 2.8           | 107.8         | 166.3      |
| BM1684 SoC  | yolov34_bmcv.py   | yolov3_int8_1b.bmodel | 3.6      | 2.8           | 59.0          | 162.7      |
| BM1684 SoC  | yolov34_bmcv.py   | yolov3_int8_4b.bmodel | 3.4      | 2.6           | 23.9          | 159.1      |
| BM1684 SoC  | yolov34_bmcv.soc  | yolov3_fp32_1b.bmodel | 5.1      | 1.5           | 102.2         | 20.1       |
| BM1684 SoC  | yolov34_bmcv.soc  | yolov3_int8_1b.bmodel | 5.1      | 1.6           | 53.6          | 20.3       |
| BM1684 SoC  | yolov34_bmcv.soc  | yolov3_int8_4b.bmodel | 4.9      | 1.5           | 19.3          | 20.0       |
| BM1684 SoC  | yolov34_sail.soc  | yolov3_fp32_1b.bmodel | 4.5      | 2.9           | 103.0         | 18.5       |
| BM1684 SoC  | yolov34_sail.soc  | yolov3_int8_1b.bmodel | 3.3      | 2.9           | 54.5          | 18.7       |
| BM1684 SoC  | yolov34_sail.soc  | yolov3_int8_4b.bmodel | 3.1      | 2.7           | 20.1          | 18.6       |
| BM1684X SoC | yolov34_opencv.py | yolov3_fp32_1b.bmodel | 3.2      | 23.1          | 163.1         | 154.0      |
| BM1684X SoC | yolov34_opencv.py | yolov3_fp16_1b.bmodel | 3.2      | 23.1          | 31.5          | 153.7      |
| BM1684X SoC | yolov34_opencv.py | yolov3_int8_1b.bmodel | 3.2      | 22.6          | 18.6          | 157.0      |
| BM1684X SoC | yolov34_opencv.py | yolov3_int8_4b.bmodel | 3.2      | 23.9          | 18.2          | 157.9      |
| BM1684X SoC | yolov34_bmcv.py   | yolov3_fp32_1b.bmodel | 3.1      | 2.2           | 159.9         | 164.2      |
| BM1684X SoC | yolov34_bmcv.py   | yolov3_fp16_1b.bmodel | 3.1      | 2.2           | 28.3          | 164.2      |
| BM1684X SoC | yolov34_bmcv.py   | yolov3_int8_1b.bmodel | 3.2      | 2.2           | 15.4          | 164.6      |
| BM1684X SoC | yolov34_bmcv.py   | yolov3_int8_4b.bmodel | 2.9      | 2.1           | 14.3          | 168.8      |
| BM1684X SoC | yolov34_bmcv.soc  | yolov3_fp32_1b.bmodel | 5.9      | 0.7           | 153.8         | 20.0       |
| BM1684X SoC | yolov34_bmcv.soc  | yolov3_fp16_1b.bmodel | 5.3      | 0.7           | 22.2          | 20.1       |
| BM1684X SoC | yolov34_bmcv.soc  | yolov3_int8_1b.bmodel | 4.4      | 0.7           | 9.3           | 20.2       |
| BM1684X SoC | yolov34_bmcv.soc  | yolov3_int8_4b.bmodel | 4.3      | 0.6           | 9.2           | 20.1       |
| BM1684X SoC | yolov34_sail.soc  | yolov3_fp32_1b.bmodel | 2.8      | 2.6           | 154.7         | 18.5       |
| BM1684X SoC | yolov34_sail.soc  | yolov3_fp16_1b.bmodel | 2.8      | 2.6           | 23.1          | 18.4       |
| BM1684X SoC | yolov34_sail.soc  | yolov3_int8_1b.bmodel | 2.8      | 2.6           | 10.2          | 18.6       |
| BM1684X SoC | yolov34_sail.soc  | yolov3_int8_4b.bmodel | 2.6      | 2.3           | 10.0          | 18.6       |
| BM1688 SoC  | yolov34_opencv.py | yolov3_fp32_1b.bmodel | 19.3     | 35.6          | 786.4         | 218.9      |
| BM1688 SoC  | yolov34_opencv.py | yolov3_fp16_1b.bmodel | 19.2     | 29.6          | 147.1         | 212.6      |
| BM1688 SoC  | yolov34_opencv.py | yolov3_int8_1b.bmodel | 19.2     | 28.7          | 43.2          | 215.8      |
| BM1688 SoC  | yolov34_opencv.py | yolov3_int8_4b.bmodel | 19.3     | 31.8          | 40.6          | 219.6      |
| BM1688 SoC  | yolov34_bmcv.py   | yolov3_fp32_1b.bmodel | 4.6      | 5.0           | 780.3         | 227.6      |
| BM1688 SoC  | yolov34_bmcv.py   | yolov3_fp16_1b.bmodel | 4.6      | 5.0           | 143.3         | 227.2      |
| BM1688 SoC  | yolov34_bmcv.py   | yolov3_int8_1b.bmodel | 4.6      | 5.0           | 39.2          | 227.8      |
| BM1688 SoC  | yolov34_bmcv.py   | yolov3_int8_4b.bmodel | 4.4      | 4.7           | 35.6          | 239.9      |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov3_fp32_1b.bmodel | 9.7      | 1.9           | 772.7         | 28.0       |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov3_fp16_1b.bmodel | 6.0      | 1.9           | 135.7         | 28.1       |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov3_int8_1b.bmodel | 6.0      | 1.9           | 31.8          | 28.3       |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov3_int8_4b.bmodel | 5.8      | 1.8           | 29.4          | 28.1       |
| BM1688 SoC  | yolov34_sail.soc  | yolov3_fp32_1b.bmodel | 7.6      | 4.9           | 774.0         | 25.7       |
| BM1688 SoC  | yolov34_sail.soc  | yolov3_fp16_1b.bmodel | 4.0      | 4.9           | 137.0         | 25.7       |
| BM1688 SoC  | yolov34_sail.soc  | yolov3_int8_1b.bmodel | 4.2      | 5.0           | 33.1          | 26.0       |
| BM1688 SoC  | yolov34_sail.soc  | yolov3_int8_4b.bmodel | 4.0      | 4.7           | 30.6          | 25.8       |
|   SRM1-20   | yolov34_opencv.py | yolov3_fp32_1b.bmodel | 13.4     | 23.8          | 342.6         | 157.1      |
|   SRM1-20   | yolov34_opencv.py | yolov3_fp16_1b.bmodel | 13.4     | 24.0          | 186.6         | 157.1      |
|   SRM1-20   | yolov34_opencv.py | yolov3_int8_1b.bmodel | 13.4     | 23.1          | 170.8         | 160.0      |
|   SRM1-20   | yolov34_opencv.py | yolov3_int8_4b.bmodel | 13.4     | 30.0          | 166.1         | 168.9      |
|   SRM1-20   | yolov34_bmcv.py   | yolov3_fp32_1b.bmodel | 24.2     | 4.6           | 321.7         | 166.2      |
|   SRM1-20   | yolov34_bmcv.py   | yolov3_fp16_1b.bmodel | 24.2     | 4.6           | 165.7         | 165.0      |
|   SRM1-20   | yolov34_bmcv.py   | yolov3_int8_1b.bmodel | 24.2     | 4.5           | 150.2         | 166.4      |
|   SRM1-20   | yolov34_bmcv.py   | yolov3_int8_4b.bmodel | 23.9     | 4.2           | 149.1         | 177.2      |
|   SRM1-20   | yolov34_bmcv.pcie | yolov3_fp32_1b.bmodel | 10.3     | 1.1           | 182.5         | 50.8       |
|   SRM1-20   | yolov34_bmcv.pcie | yolov3_fp16_1b.bmodel | 12.0     | 1.1           | 26.3          | 66.3       |
|   SRM1-20   | yolov34_bmcv.pcie | yolov3_int8_1b.bmodel | 12.4     | 1.1           | 11.0          | 66.9       |
|   SRM1-20   | yolov34_bmcv.pcie | yolov3_int8_4b.bmodel | 11.0     | 0.9           | 10.8          | 57.8       |
|   SRM1-20   | yolov34_sail.pcie | yolov3_fp32_1b.bmodel | 23.2     | 2.8           | 316.8         | 16.1       |
|   SRM1-20   | yolov34_sail.pcie | yolov3_fp16_1b.bmodel | 12.5     | 2.3           | 80.7          | 15.6       |
|   SRM1-20   | yolov34_sail.pcie | yolov3_int8_1b.bmodel | 23.2     | 2.7           | 145.0         | 16.3       |
|   SRM1-20   | yolov34_sail.pcie | yolov3_int8_4b.bmodel | 23.1     | 1.7           | 145.8         | 15.8       |

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.3，nms_thresh=0.5，yolov4性能测试结果如下：
|    测试平台  |     测试程序      |       测试模型         |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | -------- | ------------  | ---------     | ---------- |
| BM1684 SoC  | yolov34_opencv.py | yolov4_fp32_1b.bmodel | 2.9      | 2.0           | 84.2          | 10.4       |
| BM1684 SoC  | yolov34_opencv.py | yolov4_int8_1b.bmodel | 2.8      | 1.8           | 34.8          | 7.8        |
| BM1684 SoC  | yolov34_opencv.py | yolov4_int8_4b.bmodel | 2.7      | 2.0           | 20.2          | 6.0        |
| BM1684 SoC  | yolov34_bmcv.py   | yolov4_fp32_1b.bmodel | 2.7      | 2.1           | 81.6          | 10.4       |
| BM1684 SoC  | yolov34_bmcv.py   | yolov4_int8_1b.bmodel | 2.6      | 1.9           | 32.5          | 10.4       |
| BM1684 SoC  | yolov34_bmcv.py   | yolov4_int8_4b.bmodel | 2.3      | 1.6           | 16.8          | 5.4        |
| BM1684 SoC  | yolov34_bmcv.soc  | yolov4_fp32_1b.bmodel | 4.9      | 1.6           | 77.5          | 8.3        |
| BM1684 SoC  | yolov34_bmcv.soc  | yolov4_int8_1b.bmodel | 4.7      | 1.5           | 28.6          | 7.9        |
| BM1684 SoC  | yolov34_bmcv.soc  | yolov4_int8_4b.bmodel | 3.6      | 1.4           | 13.8          | 6.2        |
| BM1684 SoC  | yolov34_sail.soc  | yolov4_fp32_1b.bmodel | 4.2      | 14.2          | 80.6          | 5.4        |
| BM1684 SoC  | yolov34_sail.soc  | yolov4_int8_1b.bmodel | 4.1      | 11.8          | 31.4          | 5.1        |
| BM1684 SoC  | yolov34_sail.soc  | yolov4_int8_4b.bmodel | 3.5      | 2.6           | 15.6          | 2.7        |
| BM1684X SoC | yolov34_opencv.py | yolov4_fp32_1b.bmodel | 3.2      | 12.6          | 75.4          | 45.0       |
| BM1684X SoC | yolov34_opencv.py | yolov4_fp16_1b.bmodel | 3.2      | 13.1          | 18.5          | 45.1       |
| BM1684X SoC | yolov34_opencv.py | yolov4_int8_1b.bmodel | 3.2      | 12.4          | 11.0          | 45.0       |
| BM1684X SoC | yolov34_opencv.py | yolov4_int8_4b.bmodel | 3.2      | 14.5          | 10.2          | 47.7       |
| BM1684X SoC | yolov34_bmcv.py   | yolov4_fp32_1b.bmodel | 3.1      | 1.6           | 73.8          | 45.0       |
| BM1684X SoC | yolov34_bmcv.py   | yolov4_fp16_1b.bmodel | 3.1      | 1.6           | 17.0          | 44.6       |
| BM1684X SoC | yolov34_bmcv.py   | yolov4_int8_1b.bmodel | 3.1      | 1.6           | 9.5           | 44.4       |
| BM1684X SoC | yolov34_bmcv.py   | yolov4_int8_4b.bmodel | 3.0      | 1.5           | 8.3           | 44.9       |
| BM1684X SoC | yolov34_bmcv.soc  | yolov4_fp32_1b.bmodel | 4.3      | 0.5           | 70.4          | 8.0        |
| BM1684X SoC | yolov34_bmcv.soc  | yolov4_fp16_1b.bmodel | 4.3      | 0.5           | 13.6          | 8.0        |
| BM1684X SoC | yolov34_bmcv.soc  | yolov4_int8_1b.bmodel | 4.3      | 0.5           | 6.1           | 8.0        |
| BM1684X SoC | yolov34_bmcv.soc  | yolov4_int8_4b.bmodel | 4.2      | 0.4           | 5.6           | 7.8        |
| BM1684X SoC | yolov34_sail.soc  | yolov4_fp32_1b.bmodel | 2.8      | 1.6           | 70.9          | 7.2        |
| BM1684X SoC | yolov34_sail.soc  | yolov4_fp16_1b.bmodel | 2.7      | 1.6           | 14.1          | 7.2        |
| BM1684X SoC | yolov34_sail.soc  | yolov4_int8_1b.bmodel | 2.7      | 1.6           | 6.6           | 7.2        |
| BM1684X SoC | yolov34_sail.soc  | yolov4_int8_4b.bmodel | 2.6      | 1.4           | 6.0           | 7.1        |
| BM1688 SoC  | yolov34_opencv.py | yolov4_fp32_1b.bmodel | 19.1     | 16.0          | 380.3         | 61.8       |
| BM1688 SoC  | yolov34_opencv.py | yolov4_fp16_1b.bmodel | 19.5     | 15.6          | 99.3          | 63.0       |
| BM1688 SoC  | yolov34_opencv.py | yolov4_int8_1b.bmodel | 19.2     | 15.7          | 23.2          | 61.7       |
| BM1688 SoC  | yolov34_opencv.py | yolov4_int8_4b.bmodel | 19.3     | 18.1          | 21.0          | 64.6       |
| BM1688 SoC  | yolov34_bmcv.py   | yolov4_fp32_1b.bmodel | 4.6      | 3.9           | 378.4         | 61.9       |
| BM1688 SoC  | yolov34_bmcv.py   | yolov4_fp16_1b.bmodel | 4.6      | 3.9           | 97.2          | 62.0       |
| BM1688 SoC  | yolov34_bmcv.py   | yolov4_int8_1b.bmodel | 4.5      | 4.0           | 21.3          | 61.8       |
| BM1688 SoC  | yolov34_bmcv.py   | yolov4_int8_4b.bmodel | 4.3      | 3.6           | 18.4          | 61.8       |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov4_fp32_1b.bmodel | 5.9      | 1.4           | 374.1         | 11.2       |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov4_fp16_1b.bmodel | 5.9      | 1.4           | 92.9          | 11.2       |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov4_int8_1b.bmodel | 5.8      | 1.4           | 17.0          | 11.2       |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov4_int8_4b.bmodel | 5.7      | 1.3           | 15.2          | 11.0       |
| BM1688 SoC  | yolov34_sail.soc  | yolov4_fp32_1b.bmodel | 4.0      | 3.5           | 374.9         | 10.0       |
| BM1688 SoC  | yolov34_sail.soc  | yolov4_fp16_1b.bmodel | 4.1      | 3.7           | 93.8          | 10.1       |
| BM1688 SoC  | yolov34_sail.soc  | yolov4_int8_1b.bmodel | 4.2      | 3.6           | 17.8          | 10.1       |
| BM1688 SoC  | yolov34_sail.soc  | yolov4_int8_4b.bmodel | 4.0      | 3.3           | 15.8          | 10.0       |
|   SRM1-20   | yolov34_opencv.py | yolov4_fp32_1b.bmodel | 13.5     | 12.3          | 161.7         | 95.9       |
|   SRM1-20   | yolov34_opencv.py | yolov4_fp16_1b.bmodel | 13.4     | 12.2          | 93.2          | 95.4       |
|   SRM1-20   | yolov34_opencv.py | yolov4_int8_1b.bmodel | 13.4     | 12.2          | 81.9          | 92.2       |
|   SRM1-20   | yolov34_opencv.py | yolov4_int8_4b.bmodel | 13.3     | 17.0          | 80.1          | 97.4       |
|   SRM1-20   | yolov34_bmcv.py   | yolov4_fp32_1b.bmodel | 14.3     | 3.8           | 117.6         | 105.2      |
|   SRM1-20   | yolov34_bmcv.py   | yolov4_fp16_1b.bmodel | 23.0     | 4.2           | 78.2          | 102.7      |
|   SRM1-20   | yolov34_bmcv.py   | yolov4_int8_1b.bmodel | 12.1     | 3.7           | 33.6          | 100.2      |
|   SRM1-20   | yolov34_bmcv.py   | yolov4_int8_4b.bmodel | 24.2     | 3.9           | 74.1          | 101.7      |
|   SRM1-20   | yolov34_bmcv.pcie | yolov4_fp32_1b.bmodel | 13.8     | 0.8           | 83.6          | 37.5       |
|   SRM1-20   | yolov34_bmcv.pcie | yolov4_fp16_1b.bmodel | 12.7     | 0.8           | 16.2          | 32.5       |
|   SRM1-20   | yolov34_bmcv.pcie | yolov4_int8_1b.bmodel | 23.0     | 1.0           | 7.2           | 69.5       |
|   SRM1-20   | yolov34_bmcv.pcie | yolov4_int8_4b.bmodel | 22.8     | 0.8           | 6.6           | 68.6       |
|   SRM1-20   | yolov34_sail.pcie | yolov4_fp32_1b.bmodel | 16.9     | 2.1           | 116.4         | 5.6        |
|   SRM1-20   | yolov34_sail.pcie | yolov4_fp16_1b.bmodel | 23.6     | 2.6           | 80.4          | 5.8        |
|   SRM1-20   | yolov34_sail.pcie | yolov4_int8_1b.bmodel | 21.4     | 2.4           | 60.0          | 6.1        |
|   SRM1-20   | yolov34_sail.pcie | yolov4_int8_4b.bmodel | 13.7     | 1.3           | 30.8          | 5.1        |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BM1684/1684X SoC的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 8. FAQ
导出ONNX模型可以参考[YOLOv34_Export_Guide](./docs/YOLOv34_Export_Guide.md)。其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。