# LM_dataset
chat template里面把\<image\>标签换成image_special_token='@'\*196。196为映射图片后的维度。 image_ids=\[34\]\*196 是因为在'@'在tokenizer编码后为34.  

然后进行可能的填充，padding.  

然后对padding生成loss_mask 本质和attention_mask一样，都是针对padding token的处理。

# VLM
匹配到image_ids然后替换这196个id为clip，然后再通过一个网络映射为196维度。然后替换[34]*196。实际上是分成三部分拼接而成(image_id_before + image_ids + image_id_after)。

tensor.unfold。手动滑动窗口。  

后续训练与LLM完全一致。

