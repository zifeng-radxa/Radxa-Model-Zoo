# SAM模型导出
## 1. 准备工作
SAM模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​SAM官方开源仓库](https://github.com/facebookresearch/segment-anything)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。


## 2. 主要步骤

- 导出decoder部分模型：
SAM官方仓库提供了模型导出脚本'scripts/export.onnx_model.py'和'notebooks/onnx_model_example.ipynb'，可以直接使用它们导出onnx模型。
请按照您的需求修改`onnx_model = SamOnnxModel(sam, return_single_mask=True)`中`return_single_mask`的值。
以onnx_model_example.ipynb中的转换代码为例：

```python
    ...
    onnx_model_path = "decode_model_multi_mask.onnx"
    onnx_model = SamOnnxModel(sam, return_single_mask=False)  # return_single_mask=Flase时，将输出置信度前三的mask。return_single_mask=True时，将输出置信度最高的mask。

    # onnx_model_path = "decode_model_single_mask.onnx"
    # onnx_model = SamOnnxModel(sam, return_single_mask=True) 

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }
    
    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )    

```

- 导出auto的decoder部分模型：
使用上述decoder的onnx_model_example.ipynb代码为例，首先需要对SamOnnxModel进行修改。

- onnx.py中的SamOnnxModel修改部分：
如下，将predict_masks方法注释掉，修改成masks, scores = self.model.mask_decoder......的形式。
```
def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
        orig_im_size: torch.Tensor,
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)
        
        masks, scores = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            multimask_output=True,
        )
        
        # masks, scores = self.model.mask_decoder.predict_masks(
        #     image_embeddings=image_embeddings,
        #     image_pe=self.model.prompt_encoder.get_dense_pe(),
        #     sparse_prompt_embeddings=sparse_embedding,
        #     dense_prompt_embeddings=dense_embedding,
        # )
        if self.use_stability_score:
            scores = calculate_stability_score(
                masks, self.model.mask_threshold, self.stability_score_offset
            )

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, point_coords.shape[1])
        upscaled_masks = self.model.postprocess_masks(masks, self.input_size, orig_im_size)

```
然后只需将onnx_model_example.ipynb中的"point_coords"的size(1, 5, 2)修改为(64, 5, 2);"point_labels"中的size(1, 5)修改为(64, 5)和"mask_input"中的(1, 1)修改为(64, 1)并且onnx_model依然需要保持return_single_mask=True即可导出auto的decoder部分模型;

- 导出embedding部分：
需要您运行mata官方sam例程时，在实例化 `/segment-anything/segment_anything/build_sam.py` 中的`Class Sam()` 为`sam`后，直接导出`sam.image_encoder`。

如下例子为运行meta源码时，直接在`/segment-anything/segment_anything/predictor.py`中做`embedding`推理时导出`sam.image_encoder`。

```python
    ....
    # 在class SamPredictor的set_torch_image()函数末尾插入：
    model = self.model.image_encoder 
    input_image = torch.rand((1, 3, 1024, 1024)) # 初始化(1, 3, 1024, 1024)的输入，也可以直接输入真实图片数据
    torch.onnx.export(model, input_image,'embedding_model.onnx', verbose=True, opset_version=12) # 导出onnx
```
