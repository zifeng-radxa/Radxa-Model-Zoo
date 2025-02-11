#!/bin/bash
res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

scripts_dir=$(dirname $(readlink -f "$0"))
pushd $scripts_dir

# models
if [ ! -d "../models/onnx" ]; 
then
    mkdir -p ../models/onnx
    pushd ../models
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen2-VL/onnx.zip
    unzip onnx.zip -d ./
    popd
    echo "models download!"
else
    echo "models/onnx folder exist! Remove it if you need to update."
fi

popd