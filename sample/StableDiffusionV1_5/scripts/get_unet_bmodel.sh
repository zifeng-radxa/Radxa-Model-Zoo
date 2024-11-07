#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
outdir=../models/BM1684X/singlize/

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
size=768
b=2
if [ $1 == 'sd_turbo' ]; then
    size=1024
    b=1
fi
echo $b
function gen_unet_mlir()
{
    model_transform.py \
        --model_name unet \
        --model_def ../models/onnx_pt/singlize/unet_fp32.pt \
        --input_shapes [[$b,4,64,64],[1],[$b,77,$size]] \
        --mlir unet.mlir
}

function gen_unet_fp16bmodel()
{
    model_deploy.py \
        --mlir unet.mlir \
        --quantize BF16 \
        --chip bm1684x \
        --model unet_1684x_f16.bmodel

    mv unet_1684x_f16.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

gen_unet_mlir
gen_unet_fp16bmodel
popd