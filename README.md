# 当代人工智能：实验五

实验五：多模态情感分析。要求给定配对的文本和图像，预测对应的情感标签。



## 环境配置

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==2.0.0+cu118
- numpy==1.24.2
- matplotlib==3.7.1
- pandas==2.0.3
- transformers==4.30.2

You can simply run

```
pip install -r requirements.txt
```



## 文件结构

We select some important files for detailed description.

```
|-- code/
    |-- bert-base-uncased/
    |-- dataset/
        |-- data/  # 解析处理后的实验数据集
            |-- image_data/  # 图像数据 github单个仓库不能超过1GB，所以仓库中没有这个文件夹
            |-- test.json  # 测试集
            |-- train.json  # 训练集
            |-- translations_test.json  # id-text文本测试集
            |-- translations.json  # id-text文本训练+验证集
            |-- val.json  # 验证集
        |-- MVSA-multiple/  # 解析处理后的预训练数据集
    |-- util/
    |-- convnext.py  # convnext模型代码
    |-- data_process.py  # 加载训练数据并处理
    |-- data_process_test.py  # 加载测试数据并处理
    |-- dev_process.py  # 模型验证
    |-- get_text_label.py  # the main testing code
    |-- graph.py  # 画图代码
    |-- main.py  # the main training code
    |-- model.py  # bert和融合模型代码
    |-- pre_model.py
    |-- requirements.txt  # 依赖包
    |-- test.sh  # 测试脚本
    |-- test_without_label.txt  # 预测文件
    |-- test_process.py  # 模型测试
    |-- train.sh  # 训练脚本
    |-- train.txt  # 训练文件
    |-- train_process.py  # 模型训练
    |-- 解析数据.py
    |-- 解析数据_test.py
|-- img/
|-- README.md
```



## 执行流程

首先，进入项目目录：

```terminal
cd code
```

在目录下新建 checkpoint 文件夹。

针对不同的实验目的，需要手动更改对应的几处代码。可以按照下面的执行流程来做。

1. **多模态融合模型 + 预训练：**

在终端运行指令：

```terminal
sh train.sh 0
```

<img src="https://github.com/Silhouette0/MultiModel_Sentiment_Analysis/blob/main/img/1.png" width=700>

此时得到的结果为根据 MVSA 预训练数据集训练出的预训练模型。

训练结束后，选择 checkpoint 文件夹下效果最好的模型，更改**（main.py）**第 137 行的内容为：：

```python
	torch.load("./checkpoint/.../xxx.pth"), strict=True)  # 效果最好的模型路径
```

修改 train.sh 内的 `my_type` 为 6，再在终端运行指令：

```terminal
sh train.sh 0
```

<img src="https://github.com/Silhouette0/MultiModel_Sentiment_Analysis/blob/main/img/2.png" width=700>

此时得到的模型即为根据 MVSA 预训练模型进行微调之后的实验数据集上的训练结果。

选择其中最好的结果，更改**（get_text_label.py）**第 158 行相应代码为：

```python
	model_path = "./checkpoint/.../xxx.pth"  # 效果最好的模型路径
```

之后，在终端运行指令：

```terminal
sh test.sh 0
```

可以得到 result.txt，即为测试集 test_without_label.txt 上的情感标签预测结果。



2. **多模态融合模型 + 直接训练：**

修改 train.sh 内的 `my_type` 为 5，在终端运行指令：

```terminal
sh train.sh 0
```

<img src="https://github.com/Silhouette0/MultiModel_Sentiment_Analysis/blob/main/img/5.png" width=700>



3. **Bert：**

修改 train.sh 内的 `my_type` 为 0，在终端运行指令：

```terminal
sh train.sh 0
```

<img src="https://github.com/Silhouette0/MultiModel_Sentiment_Analysis/blob/main/img/3.png" width=700>



4. **ConvNeXt：**

修改 train.sh 内的 `my_type` 为 1，在终端运行指令：

```terminal
sh train.sh 0
```

<img src="https://github.com/Silhouette0/MultiModel_Sentiment_Analysis/blob/main/img/4.png" width=700>



5. **多模态融合模型消融实验（仅图像）：**

修改 train.sh 内的 `my_type` 为 3，在终端运行指令：

```terminal
sh train.sh 0
```



6. **多模态融合模型消融实验（仅文本）：**

修改 train.sh 内的 `my_type` 为 4，在终端运行指令：

```terminal
sh train.sh 0
```



总结一下 `my_type` 的对应关系：

```
# 0 Bert
# 1 ConvNeXt
# 2 根据MVSA-multiple数据集预训练模型
# 3 仅图像
# 4 仅文本
# 5 直接训练
# 6 根据MVSA-multiple数据集预训练出的模型进行实验数据集的训练微调
```



## 参考

Parts of this code are based on the following repositories:

- [Link-Li/CLMLF](https://github.com/Link-Li/CLMLF/tree/main)
- [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
