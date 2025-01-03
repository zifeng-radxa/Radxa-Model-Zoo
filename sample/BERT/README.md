# BERT

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
  * [4.1 TPU-NNTC编译BModel](#41-tpu-nntc编译bmodel)
  * [4.2 TPU-MLIR编译BModel](#42-tpu-mlir编译bmodel)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
  * [6.1 测试方法](#61-测试方法)
  * [6.2 测试结果](#62-测试结果)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
* [8. FAQ](#8-faq)
  
## 1. 简介
​BERT的全称为Bidirectional Encoder Representation from Transformers，是一个预训练的语言表征模型。它强调了不再像以往一样采用传统的单向语言模型或者把两个单向语言模型进行浅层拼接的方法进行预训练，而是采用新的masked language model（MLM），以致能生成深度的双向语言表征。BERT论文发表时提及在11个NLP（Natural Language Processing，自然语言处理）任务中获得了新的state-of-the-art的结果，令人惊叹。
本例程对[A simple training framework that recreates bert4keras in PyTorch. bert4torch](https://github.com/Tongjilibo/bert4torch/)的模型和算法进行移植,使之能在SOPHON BM1684\BM1684X\BM1688\CV186X上进行推理测试。
## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC、riscv PCIe)、BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32、FP16(BM1684X/BM1688/CV186X)模型编译和推理
* 支持基于sail的C++推理
* 支持基于sail的Python推理
* 支持单batch和多batch模型推理
* 支持文本测试
 
## 3. 准备模型与数据
如果您使用BM1684，建议使用TPU-NNTC编译BModel，Pytorch模型在编译前要导出成torchscript模型或onnx模型；如果您使用BM1684X，建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。具体可参考[BERT模型导出](./docs/BERT4torch_Exportonnx_Guide.md)。

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
./models/
├── BM1684
│   └── bert4torch_output_fp32_1b.bmodel     # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=1
│   └── bert4torch_output_fp32_8b.bmodel     # 使用TPU-MLIR编译，用于BM1684的FP32 BModel，batch_size=8
├── BM1684X
│   └── bert4torch_output_fp32_1b.bmodel     # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   └── bert4torch_output_fp32_8b.bmodel     # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=8
│   └── bert4torch_output_fp16_1b.bmodel     # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   └── bert4torch_output_fp16_8b.bmodel     # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=8
├── BM1688
│   └── bert4torch_output_fp32_1b.bmodel     # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1
│   └── bert4torch_output_fp32_8b.bmodel     # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=8
│   └── bert4torch_output_fp16_1b.bmodel     # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1
│   └── bert4torch_output_fp16_8b.bmodel     # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=8
│   └── bert4torch_output_fp32_1b_2core.bmodel     # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1
│   └── bert4torch_output_fp32_8b_2core.bmodel     # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=8
│   └── bert4torch_output_fp16_1b_2core.bmodel     # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1
│   └── bert4torch_output_fp16_8b_2core.bmodel     # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=8
├── CV186X
│   └── bert4torch_output_fp32_1b.bmodel     # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   └── bert4torch_output_fp32_8b.bmodel     # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=8
│   └── bert4torch_output_fp16_1b.bmodel     # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
│   └── bert4torch_output_fp16_8b.bmodel     # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=8
├── pre_train
│   └── vocab.txt
└── torch
    └── bert4torch_jit.pt
```
下载的数据包括：
```
./datasets/china-people-daily-ner-corpus                     # 人民日报数据集
├── example.dev                                              # 验证集
├── example.test                                             # 测试集
└── example.train                                            # 训练集
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

注意：mlir版本需要>=0625的版本为v1.2.2。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684/BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

​执行上述命令会在`models/BM1684/`下生成`bert4torch_output_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`bert4torch_fp16_1b.bmodel`文件，即转换好的FP16 BModel。


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试文本)或[Python例程](python/README.md#22-测试文本)推理要测试的数据集，生成预测的txt文件，注意修改数据集(datasets/china-people-daily-ner-corpus)。  
然后，使用`tools`目录下的`eval_people.py`脚本，将测试生成的txt文件与测试集标签txt文件进行对比，计算出NER的评价指标，命令如下：
```bash
# 安装seqeval，若已安装请跳过
pip3 install bert4torch==0.3.0 packaging==23.2 seqeval==1.2.2
# 请根据实际情况修改程序路径和txt文件路径, 建议在tools目录下执行该脚本，因为脚本中将工作目录设置为脚本所在目录。
cd tools
python3 eval_people.py --test_path ../datasets/china-people-daily-ner-corpus/example.test --input_path ../python/results/bert4torch_output_fp16_8b.bmodel_sail_python_result.txt
```
### 6.2 测试结果
在china-people-daily-ner-corpus数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |              测试模型               |f1             |accuary   |
| ------------ | ---------------- | ----------------------------------- | ------------- | -------- |
| SE5-16       | bert_sail.pcie   | bert4torch_output_fp32_1b.bmodel    | 0.9203        | 0.9914   |
| SE5-16       | bert_sail.pcie   | bert4torch_output_fp32_8b.bmodel    | 0.9185        | 0.9914   |
| SE5-16       | bert_sail.py     | bert4torch_output_fp32_1b.bmodel    | 0.9173        | 0.9915   |
| SE5-16       | bert_sail.py     | bert4torch_output_fp32_8b.bmodel    | 0.9163        | 0.9915   |
| SE7-32       | bert_sail.pcie   | bert4torch_output_fp32_1b.bmodel    | 0.9130        | 0.9907   |
| SE7-32       | bert_sail.pcie   | bert4torch_output_fp32_8b.bmodel    | 0.9130        | 0.9907   |
| SE7-32       | bert_sail.pcie   | bert4torch_output_fp16_1b.bmodel    | 0.9121        | 0.9907   |
| SE7-32       | bert_sail.pcie   | bert4torch_output_fp16_8b.bmodel    | 0.9120        | 0.9907   |
| SE7-32       | bert_sail.py     | bert4torch_output_fp32_1b.bmodel    | 0.9161        | 0.9914   |
| SE7-32       | bert_sail.py     | bert4torch_output_fp32_8b.bmodel    | 0.9224        | 0.9917   |
| SE7-32       | bert_sail.py     | bert4torch_output_fp16_1b.bmodel    | 0.9219        | 0.9915   |
| SE7-32       | bert_sail.py     | bert4torch_output_fp16_8b.bmodel    | 0.9191        | 0.9915   |
| SE9-16       | bert_sail.py   | bert4torch_output_fp32_1b.bmodel |   0.9154 |   0.9915 |
| SE9-16       | bert_sail.py   | bert4torch_output_fp32_8b.bmodel |   0.9211 |   0.9915 |
| SE9-16       | bert_sail.py   | bert4torch_output_fp16_1b.bmodel |   0.9200 |   0.9915 |
| SE9-16       | bert_sail.py   | bert4torch_output_fp16_8b.bmodel |   0.9176 |   0.9915 |
| SE9-16       | bert_sail.soc  | bert4torch_output_fp32_1b.bmodel |   0.9130 |   0.9908 |
| SE9-16       | bert_sail.soc  | bert4torch_output_fp32_8b.bmodel |   0.9130 |   0.9908 |
| SE9-16       | bert_sail.soc  | bert4torch_output_fp16_1b.bmodel |   0.9129 |   0.9908 |
| SE9-16       | bert_sail.soc  | bert4torch_output_fp16_8b.bmodel |   0.9129 |   0.9908 |
| SE9-16       | bert_sail.py   | bert4torch_output_fp32_1b_2core.bmodel |   0.9183 |   0.9916 |
| SE9-16       | bert_sail.py   | bert4torch_output_fp32_8b_2core.bmodel |   0.9183 |   0.9915 |
| SE9-16       | bert_sail.py   | bert4torch_output_fp16_1b_2core.bmodel |   0.9187 |   0.9915 |
| SE9-16       | bert_sail.py   | bert4torch_output_fp16_8b_2core.bmodel |   0.9211 |   0.9916 |
| SE9-16       | bert_sail.soc  | bert4torch_output_fp32_1b_2core.bmodel |   0.9130 |   0.9908 |
| SE9-16       | bert_sail.soc  | bert4torch_output_fp32_8b_2core.bmodel |   0.9130 |   0.9908 |
| SE9-16       | bert_sail.soc  | bert4torch_output_fp16_1b_2core.bmodel |   0.9129 |   0.9908 |
| SE9-16       | bert_sail.soc  | bert4torch_output_fp16_8b_2core.bmodel |   0.9129 |   0.9908 |
| SE9-8        | bert_sail.soc    | bert4torch_output_fp32_1b.bmodel    | 0.9129        | 0.9907   |
| SE9-8        | bert_sail.soc    | bert4torch_output_fp32_8b.bmodel    | 0.9129        | 0.9907   |
| SE9-8        | bert_sail.soc    | bert4torch_output_fp16_1b.bmodel    | 0.9129        | 0.9907   |
| SE9-8        | bert_sail.soc    | bert4torch_output_fp16_8b.bmodel    | 0.9129        | 0.9907   |
| SE9-8        | bert_sail.py     | bert4torch_output_fp32_1b.bmodel    | 0.9168        | 0.9914   |
| SE9-8        | bert_sail.py     | bert4torch_output_fp32_8b.bmodel    | 0.9192        | 0.9916   |
| SE9-8        | bert_sail.py     | bert4torch_output_fp16_1b.bmodel    | 0.9199        | 0.9916   |
| SE9-8        | bert_sail.py     | bert4torch_output_fp16_8b.bmodel    | 0.9219        | 0.9916   |
| SRM1-20      | bert_sail.pcie | bert4torch_output_fp32_1b.bmodel |   0.9130 |   0.9908 |
| SRM1-20      | bert_sail.pcie | bert4torch_output_fp32_8b.bmodel |   0.9130 |   0.9908 |
| SRM1-20      | bert_sail.pcie | bert4torch_output_fp16_1b.bmodel |   0.9122 |   0.9907 |
| SRM1-20      | bert_sail.pcie | bert4torch_output_fp16_8b.bmodel |   0.9121 |   0.9908 |
| SRM1-20      | bert_sail.py     | bert4torch_output_fp32_1b.bmodel    | 0.9178        | 0.9915   |
| SRM1-20      | bert_sail.py     | bert4torch_output_fp32_8b.bmodel    | 0.9217        | 0.9918   |
| SRM1-20      | bert_sail.py     | bert4torch_output_fp16_1b.bmodel    | 0.9193        | 0.9915   |
| SRM1-20      | bert_sail.py     | bert4torch_output_fp16_8b.bmodel    | 0.9162        | 0.9915   |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；
## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/bert4torch_output_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每条文本的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                        | calculate time(ms) |
| -------------------------------------------      | ----------------- |
| BM1684/bert4torch_output_fp32_1b.bmodel          | 170.848           |
| BM1684/bert4torch_output_fp32_8b.bmodel          | 146.653           |
| BM1684X/bert4torch_output_fp32_1b.bmodel         | 91.473            |
| BM1684X/bert4torch_output_fp16_1b.bmodel         | 8.643             |
| BM1684X/bert4torch_output_fp32_8b.bmodel         | 87.478            |
| BM1684X/bert4torch_output_fp16_8b.bmodel         | 5.726             |
| BM1688/bert4torch_output_fp32_1b.bmodel          | 269.7             |
| BM1688/bert4torch_output_fp16_1b.bmodel          | 40.0              |
| BM1688/bert4torch_output_fp32_8b.bmodel          | 263.5             |
| BM1688/bert4torch_output_fp16_8b.bmodel          | 34.1              |
| BM1688/bert4torch_output_fp32_1b_2core.bmodel    | 184.3             |
| BM1688/bert4torch_output_fp16_1b_2core.bmodel    | 30.2              |
| BM1688/bert4torch_output_fp32_8b_2core.bmodel    | 156.8             |
| BM1688/bert4torch_output_fp16_8b_2core.bmodel    | 18.5              |
| CV186X/bert4torch_output_fp16_1b.bmodel|          39.14  |
| CV186X/bert4torch_output_fp32_1b.bmodel|         270.40  |
| CV186X/bert4torch_output_fp16_8b.bmodel|         33.03   |
| CV186X/bert4torch_output_fp32_8b.bmodel|         264.05  |
> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每条文本的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/china-people-daily-ner-corpus/example.test`，性能测试结果如下：
|    测试平台  |     测试程序      |             测试模型                |tot_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- |  --------- | ---------   | ---------   |
| SE5-16      | bert_sail.py     | bert4torch_output_fp32_1b.bmodel    | 307.64   | 3.48       | 171.87      | 132.26      |
| SE5-16      | bert_sail.py     | bert4torch_output_fp32_8b.bmodel    | 168.89   | 3.46       | 147.06      | 18.359      |
| SE5-16      | bert_sail.soc    | bert4torch_output_fp32_1b.bmodel    | 19.52    | 19.11      | 0.35        | 0.022       |
| SE5-16      | bert_sail.soc    | bert4torch_output_fp32_8b.bmodel    | 20.21    | 19.34      | 0.830       | 0.021       |
| SE7-32      | bert_sail.py     | bert4torch_output_fp32_1b.bmodel    | 225.16   | 3.50       | 92.25       | 129.39      |
| SE7-32      | bert_sail.py     | bert4torch_output_fp32_8b.bmodel    | 109.60   | 3.5        | 87.76       | 18.36       |
| SE7-32      | bert_sail.py     | bert4torch_output_fp16_1b.bmodel    | 141.59   | 3.5        | 9.50        | 128.57      |
| SE7-32      | bert_sail.py     | bert4torch_output_fp16_8b.bmodel    | 27.64    | 3.4        | 5.84        | 18.325      |
| SE7-32      | bert_sail.soc    | bert4torch_output_fp32_1b.bmodel    | 19.45    | 19.14      | 0.028       | 0.022       |
| SE7-32      | bert_sail.soc    | bert4torch_output_fp32_8b.bmodel    | 19.28    | 19.15      | 0.078       | 0.021       |
| SE7-32      | bert_sail.soc    | bert4torch_output_fp16_1b.bmodel    | 19.87    | 19.59      | 0.218       | 0.020       |
| SE7-32      | bert_sail.soc    | bert4torch_output_fp16_8b.bmodel    | 19.73    | 19.62      | 0.642       | 0.019       |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp32_1b.bmodel|     473.89      |      4.92       |     274.80      |     194.11      |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp32_8b.bmodel|     297.54      |      4.90       |     264.84      |      27.79      |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp16_1b.bmodel|     243.83      |      4.90       |      41.54      |     197.32      |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp16_8b.bmodel|      67.42      |      4.94       |      34.47      |      28.00      |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp32_1b.bmodel|     298.32      |      27.11      |     271.15      |      0.04       |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp32_8b.bmodel|     291.28      |      26.79      |     264.45      |      0.03       |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp16_1b.bmodel|      67.56      |      26.86      |      40.66      |      0.04       |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp16_8b.bmodel|      61.32      |      26.96      |      34.33      |      0.03       |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp32_1b_2core.bmodel|     381.68      |      4.90       |     180.75      |     195.95      |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp32_8b_2core.bmodel|     185.70      |      4.98       |     152.89      |      27.83      |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp16_1b_2core.bmodel|     234.26      |      4.92       |      31.72      |     197.54      |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp16_8b_2core.bmodel|      51.49      |      4.89       |      18.75      |      27.84      |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp32_1b_2core.bmodel|     206.84      |      26.88      |     179.91      |      0.04       |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp32_8b_2core.bmodel|     179.37      |      26.73      |     152.61      |      0.03       |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp16_1b_2core.bmodel|      57.63      |      26.82      |      30.77      |      0.03       |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp16_8b_2core.bmodel|      45.67      |      27.01      |      18.64      |      0.03       |
|    SE9-8    |   bert_sail.soc   |bert4torch_output_fp32_1b.bmodel|     297.99      |      27.09      |     270.85      |      0.04       |
|    SE9-8    |   bert_sail.soc   |bert4torch_output_fp32_8b.bmodel|     290.67      |      26.48      |     264.16      |      0.03       |
|    SE9-8    |   bert_sail.soc   |bert4torch_output_fp16_1b.bmodel|      66.50      |      27.02      |      39.45      |      0.03       |
|    SE9-8    |   bert_sail.soc   |bert4torch_output_fp16_8b.bmodel|      60.02      |      26.85      |      33.13      |      0.03       |
|    SE9-8    |   bert_sail.py    |bert4torch_output_fp32_1b.bmodel|     472.14      |      4.88       |     272.00      |     195.19      |
|    SE9-8    |   bert_sail.py    |bert4torch_output_fp32_8b.bmodel|     297.02      |      4.89       |     264.52      |      27.60      |
|    SE9-8    |   bert_sail.py    |bert4torch_output_fp16_1b.bmodel|     242.85      |      4.89       |      40.59      |     197.30      |
|    SE9-8    |   bert_sail.py    |bert4torch_output_fp16_8b.bmodel|      65.86      |      4.92       |      33.27      |      27.66      |
|   SRM1-20   |   bert_sail.pcie  |bert4torch_output_fp32_1b.bmodel|     125.78      |      12.78      |     112.95      |      0.04       |
|   SRM1-20   |   bert_sail.pcie  |bert4torch_output_fp32_8b.bmodel|     119.95      |      12.59      |     107.36      |      0.03       |
|   SRM1-20   |   bert_sail.pcie  |bert4torch_output_fp16_1b.bmodel|      25.00      |      12.88      |      12.07      |      0.04       |
|   SRM1-20   |   bert_sail.pcie  |bert4torch_output_fp16_8b.bmodel|      19.64      |      12.37      |       7.25      |      0.03       |
|   SRM1-20   |   bert_sail.py    |bert4torch_output_fp32_1b.bmodel|     435.81      |      3.17       |     113.62      |     318.98      |
|   SRM1-20   |   bert_sail.py    |bert4torch_output_fp32_8b.bmodel|     159.42      |      3.12       |     107.54      |     48.76       |
|   SRM1-20   |   bert_sail.py    |bert4torch_output_fp16_1b.bmodel|     340.37      |      3.28       |      12.84      |     324.20      |
|   SRM1-20   |   bert_sail.py    |bert4torch_output_fp16_8b.bmodel|      56.26      |      3.50       |       7.36      |      45.39      |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每个文本处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. BERT cpp的后处理只有softmax，耗时很短，可以忽略。
## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
