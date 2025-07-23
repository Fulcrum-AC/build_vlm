# 从零开始搭建多模态大模型
## 1. 简介
使用预训练的LLM与视觉编码器作为基础组件并进行适当改写，以支持多模态的输入；编写了projector层，将来自视觉编码器的图像特征映射到LLM支持的输入维度，实现模态对齐。
- 数据集：[minimind-v_dataset](https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset/files"点击跳转")
- 视觉编码器：[google/siglip-base-patch16-224](https://hf-mirror.com/google/siglip-base-patch16-224"点击跳转")
- 语言模型：[Qwen/Qwen2.5-0.5B](https://hf-mirror.com/Qwen/Qwen2.5-0.5B"点击跳转")
## 2. 实现流程
### 2.1. 环境搭建
采用如下命令安装项目所需依赖
```
# 安装PyTorch nightly (支持CUDA 12.8)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 安装所需的库
pip install transformers peft accelerate
```
### 2.2. 数据处理
本项目使用的数据集质量较高，无需清洗，只需处理成对话格式即可。
原始数据如下：
```
{"conversations": [{"role": "user", "content": "绘制清晰简洁的图片摘要.\n<image>"}, {"role": "assistant", "content": "西拉穿着这件上衣和这件裙子看起来很漂亮,当时她和以前大学的同学一起表演"}], "image": "GCC_train_000000001.jpg"}
```
处理成如下的对话格式：
```
<|im_start|>user绘制清晰简洁的图片摘要.<|vision_start|><|vision_end|><|endoftext|><|im_end|>
<|im_start|>assistant西拉穿着这件上衣和这件裙子看起来很漂亮,当时她和以前大学的同学一起表演<|endoftext|><|im_end|>
```
<|im_start|>，<|im_end|>，<|endoftext|>，<|vision_start|>，<|vision_end|>，user，assistant均为特殊token，除user，assistant之外，其余均为分词器预训练的special tokens。后续将196个图像特征添加到<|im_start|>和<|im_end|>两个token之间。

图像数据通过预训练的image_processor处理成pixel_values，形状为（batch_size,channels,height,width）
### 2.3. 模型搭建
基本思路：

1. 编写一个VLM类，集成了视觉编码器、LLM、映射层，并包含了分词、图像文本拼接、标签构建、掩码构建等功能。
2. 取视觉编码器最后一个隐藏状态作为图像特征，张量形状为（batch_size,num_patches,hidden_dim），经过映射层后得到对齐的图像特征aligned_image_features.
3. 将文本进行分词得到input_ids，采取左填充方式，长度设置为32，之后与aligned_image_features一起输入LLM类的实例。
4. 在LLM类中，首先将input_ids进行嵌入，得到嵌入向量text_embeds;将text_embeds与aligned_image_features进行拼接，同时要改变padding_mask的长度，并手动构建因果掩码casual_mask,保证prompt部分不能看到答案，答案部分只能看到当前输出之前的内容。将padding_mask与causal_mask进行叠加，得到最终的combined_mask;此外，在LLM中还要实现labels的手动构建，保证序列中除了assistant token至<|im_end|>这段序列之外，其余部分均设置为-100。
### 2.4. 微调
冻结视觉编码器，并对映射层进行全量微调，对LLM以LoRA方式进行参数高效微调。

训练时的参数设置如下：
```python
train_args = TrainingArguments(
        output_dir="./vlm_model_output5",
        per_device_train_batch_size=40,  # 设置训练批大小
        save_strategy="steps",        # 按步数保存
        save_steps=500,               # 每500步保存一次检查点
        save_total_limit=20,           # 只保留最新的20个检查点
        logging_strategy="steps",     # 按步数记录日志
        logging_steps=10,            # 每10步记录一次日志
        bf16=True, 
        learning_rate=3e-4,  
        num_train_epochs=8,
        weight_decay=0.01,    # 添加权重衰减防止过拟合
        warmup_ratio=0.02,    # 总步数的2%用于预热
        lr_scheduler_type="cosine",  # 余弦退火
        max_grad_norm=1.0,     # 梯度裁剪
        save_safetensors=False # 禁用Safetensors格式
    )
```
### 3. 测试效果
| 图片 | 描述  | 
| :--- | ---- | 
| <img width="128" height="128" alt="image" src="https://github.com/user-attachments/assets/e878fc50-b313-418f-a83b-ae684bd1775f" />|     |


<table>
  <thead>
    <tr>
      <th>图片</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <img src="./dataset/eval_images/城市车水马龙-city-traffic.jpg" alt="city-traffic">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>图中是一个繁忙的城市街道，一条长长的街道两旁都是高楼大厦。这条街上挤满了汽车、卡车和公共汽车，还有许多其他车辆在路上行驶。在街道上，可以看到许多汽车，有的在高速行驶，而其他的则停在街道一侧。此外还有一辆公交车也停在街道的右侧。街道上可以看到交通灯，表明这是一个繁忙的城市环境。</td>
      <td>图中是一个繁忙的城市景象，有几辆汽车和一辆卡车行驶在城市街道上。可以看到许多交通信号灯，其中一些位于街道左侧，另一些则在右侧。可以看到有几个人在街上行走，其中一些人站得离街道更近一些，而另一些则距离较远。还有一个停车标志位于画面的左侧，暗示着城市环境。可以看到街道上有两辆汽车，一辆在右边，另一辆在左边，还有一辆在左边。这幅图像捕捉到了都市环境中典型的一天。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/太空宇航员-Astronaut-Space.jpg" alt="astronaut">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>图片显示了一个宇航员的宇航员身穿宇航服，坐在一架大型航天飞机上。他们似乎正在进行一次宇航员登机或下机的旅程。在宇航员的身后，有一个火箭发射架，可能是用来支撑宇航员在旅程中的任务。此外，还有一架飞机停在机库附近，进一步表明这是一次航空展。在飞机的周围，还有一些人，但他们看起来离飞机很近。可以看到一个人站在飞机附近，可能正在观察或等待航天飞机准备起飞。</td>
      <td>场景中，一名士兵戴着头盔站在一架大型飞机上。这架飞机似乎是一架军用军用飞机，似乎正准备登上一架飞机。另一个人则站在前面，可能正在观察飞行过程。在飞机周围，有几个人，其中一些站在左侧，另一些则站在右侧。他们似乎正在观看飞行员的表现。此外，还有一辆卡车停在靠近左侧的位置，可能是为了更具体地观察飞行过程。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/小狗美女海边-Dog-Woman-Sea.jpg" alt="dog-woman-sea">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>图片中，一个女人坐在沙滩上，手里拿着一只白色的狗。她看起来像是个女人，坐在沙地上，看着她。一只狗也坐在她旁边，看起来很放松和舒适。海滩上散布着其他沙滩游客，有些人坐着，而另一些人则坐在更远的地方。背景中可以看到一艘船，这表明这是一个受欢迎的海滩旅游目的地。</td>
      <td>两个人坐在海滩上，一边懒洋洋地躺在沙滩上，另一边则坐着。他们似乎正在享受海边时光。海滩上有几把椅子，其中一把靠近沙滩的左侧，另一把在中间。此外，还有一只狗躺在沙地上，为这个场景增添了一种放松的气氛。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/彩虹瀑布-Rainbow-Falls.jpg" alt="rainbow-falls">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>照片捕捉到一幅美丽如画的大自然场景，背景是高山峦崖。在水边，一座巨大的喷泉横跨着水面，吸引着许多游客。水面上有几个人，他们或站或坐在喷泉周围，或站或坐。有些人可以看到他们在水中行走，而其他人则站在水边。总体而言，这幅画描绘的是一个美丽而宁静的环境，在那里人们可以欣赏到如画般的美景。</td>
      <td>在一个美丽的蓝色天空下，一座巨大而巨大的白色瀑布上方悬挂着一只巨大的湿流水。这只瀑布位于一座山上，为整个场景增添了一种迷人而又宁静的气氛。在这幅图像的背景中，可以看到几艘船，其中一些靠近水边，其他的则离得较远。这些船只似乎正在为风景或户外活动做准备。</td>
    </tr>
    <tr>
      <td>
        <img src="./dataset/eval_images/椅子老人看书-Chair-Elderly-Reading.jpg" alt="elderly-reading">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>图中，一个男人坐在公园的长椅上，旁边是一把绿色椅子。他身边有一本打开的书，上面写着"读书"一句话，暗示他可能正在阅读。公园里有一张长椅和一张公园长椅，为周围的环境增添了几分生气。在公园的周围，有几辆汽车和一辆卡车，表明这是一个公共区域。此外，还可以看到一个人站在公园的不同位置上，可能是等着上路或过马路。</td>
      <td>一个穿着短裤的老人坐在公园长椅上，周围是树木。他似乎正在读一本书，可能是在读书。背景中有一座长凳，为这个场景提供了充足的座位。在背景中，可以看到一把椅子和一张餐桌，这说明这个场景可能是在一个户外座位区，那里有椅子供人们坐下来放松。</td>
    </tr>
  </tbody>
</table>
