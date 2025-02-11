# C++例程

## 目录

- [C++例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 编译程序](#2-编译程序)
    - [2.1 x86/arm/riscv PCIe平台](#21-x86armriscv-PCIe平台)
    - [2.2 SoC平台](#22-SoC平台)
  - [3. 推理测试](#2-推理测试)
    - [3.1 参数说明](#31-参数说明)
    - [3.2 测试图片](#32-测试图片)

cpp目录下提供了C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | clip_opencv   | 使用SOPHON-OpenCV解码、SOPHON-OpenCV前处理、BMRT推理   |
| 2    | chinese_clip_opencv   | CLIP模型的中文版本，使用SOPHON-OpenCV解码、SOPHON-OpenCV前处理、BMRT推理   |

## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台
如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv（对应BM1684&BM1684x SDK>=v24.04.01，BM1688&CV186AH SDK>=v1.7.0），具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包。

## 2. 编译程序
C++程序运行前需要编译可执行文件，下面以clip_opencv为例子。
### 2.1 x86/arm/riscv PCIe平台
可以直接在PCIe平台上编译程序：

```bash
cd cpp/clip_opencv
mkdir build && cd build
cmake ..
make
cd ..
```
编译完成后，会在clip_opencv目录下生成clip_opencv.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp/clip_opencv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make
```
编译完成后，会在clip_opencv目录下生成clip_opencv.soc。

- 如果在使用或者测试过程中遇到问题，可以先参考[常见问题说明](../docs/FAQ.md)

## 3. 推理测试

### 3.1 参数说明

```bash
usage: clip_opencv.pcie  [params]

        --image_path (value:../../datasets)
                sampled test case
        --text (value:"a diagram, a dog, a car")
                input text
        --dev_id (value:0)
                TPU device id
        --image_model (value:../../models/BM1684X/clip_fp32_1b.bmodel)
                image_model image_model path
        --text_model (value:../../models/BM1684X/clip_text_vitb32_bm1684x_f16_1b.bmodel)
                text_model text_model path
        --help (value:true)
                print help information.
```
**注意：** CPP传参与python不同，需要用等于号，例如`./clip_opencv.pcie --bmodel=xxx`。

### 3.2 测试图片
图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
./clip_opencv.pcie --image_path=../../datasets --text="a diagram, a dog, a car" --dev_id=0 --image_model="../../models/BM1684X/clip_image_vitb32_bm1684x_f16_1b.bmodel" --text_model="../../models/BM1684X/clip_text_vitb32_bm1684x_f16_1b.bmodel"
```
程序运行结束后，会在命令行中打印信息，输出图片和文本的匹配度。

```
Total Similarity per Image:
Filename: ../../datasets/Clothes-and-hats-misidentified-as-safety-helmet.jpg
Open /dev/bm-sophon0 successfully, device index = 0, jpu fd = 15, vpp fd = 15
Text: a dog, Similarity: 0.617956
Text: a car, Similarity: 0.318094
Text: a diagram, Similarity: 0.0639502
Filename: ../../datasets/CLIP.png
Text: a diagram, Similarity: 0.993848
Text: a car, Similarity: 0.00474531
Text: a dog, Similarity: 0.0014063
Filename: ../../datasets/Car-headlights-misidentified-as-flames.jpg
Text: a car, Similarity: 0.992424
Text: a dog, Similarity: 0.00561061
Text: a diagram, Similarity: 0.00196498

Total Similarity per Text:
Text: a diagram
Image: ../../datasets/CLIP.png, Similarity: 0.998904
Image: ../../datasets/Car-headlights-misidentified-as-flames.jpg, Similarity: 0.00108089
Image: ../../datasets/Clothes-and-hats-misidentified-as-safety-helmet.jpg, Similarity: 1.52948e-05
Text: a dog
Image: ../../datasets/Car-headlights-misidentified-as-flames.jpg, Similarity: 0.664068
Image: ../../datasets/CLIP.png, Similarity: 0.304132
Image: ../../datasets/Clothes-and-hats-misidentified-as-safety-helmet.jpg, Similarity: 0.0318009
Text: a car
Image: ../../datasets/Car-headlights-misidentified-as-flames.jpg, Similarity: 0.991202
Image: ../../datasets/CLIP.png, Similarity: 0.00865985
Image: ../../datasets/Clothes-and-hats-misidentified-as-safety-helmet.jpg, Similarity: 0.000138133
-------------------Image num 3, Preprocess average time ------------------------
preprocess(ms): 2.80289
------------------ Image num 3, Image Encoding average time ----------------------
image_encode(ms): 6.79703
------------------ Image num 3, Text Encoding average time ----------------------
text_encode(ms): 9.12867
```

中文版CLIP图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
./clip_opencv.pcie --image_path=../../datasets --text="流程图,狗,车" --dev_id=0 --image_model="../../models/BM1684X/cn_clip_image_vitb16_bm1684x_f16_1b.bmodel" --text_model="../../models/BM1684X/cn_clip_text_vitb16_bm1684x_f16_1b.bmodel"
```
程序运行结束后，会在命令行中打印信息，输出图片和文本的匹配度。

```
Total Similarity per Image:
Filename: ../../datasets/Clothes-and-hats-misidentified-as-safety-helmet.jpg
Open /dev/bm-sophon0 successfully, device index = 0, jpu fd = 15, vpp fd = 15
Text: 车, Similarity: 0.650535
Text: 流程图, Similarity: 0.185463
Text: 狗, Similarity: 0.164001
Filename: ../../datasets/CLIP.png
Text: 流程图, Similarity: 0.958328
Text: 车, Similarity: 0.026968
Text: 狗, Similarity: 0.014704
Filename: ../../datasets/Car-headlights-misidentified-as-flames.jpg
Text: 车, Similarity: 0.998814
Text: 狗, Similarity: 0.00118144
Text: 流程图, Similarity: 4.89722e-06

Total Similarity per Text:
Text: 流程图
Image: ../../datasets/CLIP.png, Similarity: 0.999516
Image: ../../datasets/Clothes-and-hats-misidentified-as-safety-helmet.jpg, Similarity: 0.000300758
Image: ../../datasets/Car-headlights-misidentified-as-flames.jpg, Similarity: 0.000183196
Text: 狗
Image: ../../datasets/Car-headlights-misidentified-as-flames.jpg, Similarity: 0.739087
Image: ../../datasets/CLIP.png, Similarity: 0.256466
Image: ../../datasets/Clothes-and-hats-misidentified-as-safety-helmet.jpg, Similarity: 0.00444761
Text: 车
Image: ../../datasets/Car-headlights-misidentified-as-flames.jpg, Similarity: 0.99922
Image: ../../datasets/CLIP.png, Similarity: 0.000752204
Image: ../../datasets/Clothes-and-hats-misidentified-as-safety-helmet.jpg, Similarity: 2.82125e-05
-------------------Image num 3, Preprocess average time ------------------------
preprocess(ms): 3.51843
------------------ Image num 3, Image Encoding average time ----------------------
image_encode(ms): 14.7889
------------------ Image num 3, Text Encoding average time ----------------------
text_encode(ms): 3.8995
```

