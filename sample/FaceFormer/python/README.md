# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 使用方式](#22-使用方式)


python目录下提供了一个Python例程，具体情况如下：

| 序号 |  Python例程       | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | faceformer.py     | 使用SAIL推理 |


## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您还需要安装其他第三方库：
```bash
sudo apt-get update
sudo apt-get install libsndfile1
pip3 install -r python/requirements.txt
```
您还需要安装sophon-sail，参考[sophon-sail编译安装指南](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#id11)自己编译sophon-sail。

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

此外您还需要安装其他第三方库：
```bash
sudo apt-get update
sudo apt-get install libsndfile1
pip3 install -r python/requirements.txt
```
由于本例程需要的sophon-sail版本较新，这里提供一个可用的sophon-sail whl包，SoC环境可以通过下面的命令下载：
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/FaceFormer/sophon_arm-3.9.0-py3-none-any.whl #arm soc, py38
pip3 install sophon_arm-3.9.0-py3-none-any.whl --force-reinstall
```
如果您需要其他版本的sophon-sail，可以参考上一小节，下载源码自己编译。

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明

```bash
usage: python3 faceformer.py [--bmodel BMODEL] [--wav_path wav] [--dataset dataset] [--dev_id DEV_ID]
--bmodel: 用于推理的bmodel路径；
--model_name：模型的名字；
--wav_path：测试的语音路径；
--dataset：测试的数据集名字；
--dev_id: 用于推理的tpu设备id；
--help: 输出帮助信息
```

### 2.2 使用方式

```bash
cd python
python3 faceformer.py --bmodel ../models/BM1684X/faceformer_f32.bmodel --model_name vocaset --wav_path ../datasets/wav/test1.wav --dataset vocaset --dev_id 0 
```

在程序执行完成后，会输出运行时间、结果以及结果的维度： “result.shape:  (XXX, 15069)”。
