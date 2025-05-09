# 中文到英文神经机器翻译系统

这个项目实现了一个基于注意力机制的序列到序列（Seq2Seq）神经网络模型，用于中文到英文的机器翻译任务。模型使用了编码器-解码器架构，并结合了Bahdanau注意力机制以提高翻译质量。

## 目录

- [项目简介](#项目简介)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [数据处理](#数据处理)
- [模型架构](#模型架构)
- [训练过程](#训练过程)
- [评估和可视化](#评估和可视化)
- [命令行参数](#命令行参数)
- [使用示例](#使用示例)

## 项目简介

本项目基于PyTorch实现了一个神经机器翻译系统，用于将中文翻译成英文。主要特点包括：

- 使用GRU（门控循环单元）作为循环神经网络的基础结构
- 实现了Bahdanau注意力机制以处理长句子
- 支持预训练分词器（GPT-2用于英文，BERT用于中文）
- 使用束搜索（Beam Search）进行解码以提高翻译质量
- 实现了早停机制，避免过拟合
- 提供了注意力权重可视化功能

## 环境要求

项目依赖以下Python库：

```
torch>=1.7.0
numpy<2.0.0
matplotlib>=3.3.0
scikit-learn>=0.23.0
tqdm>=4.50.0
transformers>=4.5.0
pandas>=1.0.0
```

## 安装步骤

1. 克隆本仓库：

```bash
git clone https://github.com/zjzjy/Chinese-to-English-seq2seq.git
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 准备数据：

确保`cmn-eng/cmn.txt`文件存在，该文件包含中文和英文的平行语料。

## 数据处理

### 数据格式

训练数据采用tab分隔的文本文件，格式为：

```
英文句子<tab>中文句子<tab>版权信息
```

例如：

```
I know none of them.    他们中的任何一位我都不认识。    CC-BY 2.0 (France) Attribution: tatoeba.org #260859 (CK) & #1446777 (asosan)
```

对于汉译英任务，我们将中文句子作为输入，英文句子作为目标输出。

### 数据预处理

数据处理流程包括：

1. **文本标准化**：对英文进行小写转换和特殊字符处理；对中文进行标点符号处理。
2. **句子长度过滤**：删除长度超过设定阈值的句子对（默认为15个词元）。
3. **分词处理**：
   - 中文（输入）：使用BERT中文分词器
   - 英文（输出）：使用GPT-2分词器
4. **数据集划分**：按照9:1的比例分割训练集和测试集。

### 预训练分词器

项目支持使用Hugging Face Transformers库中的预训练分词器：

- 中文：使用BERT中文分词器处理输入语言
- 英文：使用GPT-2分词器处理输出语言

预训练分词器可以更好地处理词汇表外(OOV)的单词，并提供更高质量的词向量表示。

## 模型架构

### 编码器 (EncoderRNN)

编码器使用GRU网络将输入的中文序列编码为上下文向量：

1. 单词嵌入层：将输入词ID转换为密集向量表示
2. GRU层：处理序列信息，生成隐藏状态
3. Dropout层：防止过拟合

### 注意力机制 (BahdanauAttention)

实现了Bahdanau注意力机制，计算解码器每一步与编码器输出的相关性：

1. 使用前馈神经网络计算注意力分数
2. 使用softmax得到注意力权重
3. 计算加权上下文向量

### 解码器 (AttnDecoderRNN)

解码器结合注意力机制生成英文序列：

1. 单词嵌入层：将目标词ID转换为密集向量
2. 注意力层：计算当前时间步的上下文向量
3. GRU层：结合上下文向量和上一时间步输出生成新的隐藏状态
4. 输出层：将隐藏状态映射为词汇表大小的分布
5. Dropout层：防止过拟合

### 束搜索解码

为了提高翻译质量，模型在推理阶段使用束搜索算法：

1. 在每个时间步保留最可能的k个序列（beam size）
2. 对每个候选序列计算累积概率
3. 选择累积概率最高的序列作为最终输出

## 训练过程

### 训练设置

- 优化器：Adam
- 损失函数：负对数似然（NLLLoss），忽略填充标记
- 批次大小：默认32/64（调试/正常模式）
- 隐藏层大小：默认256/512（调试/正常模式）
- 训练轮次：默认20/30（调试/正常模式）

### 早停机制

为避免过拟合，模型实现了早停机制：

- 如果连续多个epoch（默认5个）损失不再显著下降（默认阈值0.001），则提前终止训练
- 可通过命令行参数`--patience`和`--min_delta`调整早停条件

### 梯度裁剪

为避免梯度爆炸问题，训练过程中应用了梯度裁剪（默认阈值1.0）。

## 评估和可视化

### 评估指标

模型使用以下方式评估翻译质量：

1. 随机从测试集抽取样本进行翻译
2. 比较模型输出与参考翻译
3. 将评估结果保存为CSV文件

### 注意力权重可视化

模型可视化注意力权重，帮助理解模型如何关注输入句子的不同部分：

1. 将注意力权重绘制为热图
2. X轴表示输入中文字符
3. Y轴表示输出英文单词
4. 颜色深浅表示注意力权重大小

### 损失曲线

训练过程中的损失变化被记录并绘制为曲线图，用于监控训练进度和收敛性。

## 命令行参数

模型提供了丰富的命令行参数来控制训练和评估过程：

### 基本参数
- `--debug`：调试模式，使用较少的数据和轮次进行快速测试
- `--data_path`：数据文件路径，默认为'cmn-eng/cmn.txt'
- `--pretrained_dir`：预训练模型目录，默认为'./pretrained_models'
- `--model_path`：模型保存/加载路径，默认为'translation_model.pt'

### 模型参数
- `--hidden_size`：隐藏层大小，默认在调试/正常模式下为256/512
- `--dropout`：Dropout比率，默认为0.1
- `--max_length`：最大句子长度，默认为15

### 训练参数
- `--n_epochs`：训练轮数，默认在调试/正常模式下为20/30
- `--batch_size`：批次大小，默认在调试/正常模式下为32/64
- `--learning_rate`：学习率，默认为0.001
- `--beam_size`：束搜索大小，默认为3
- `--sample_size`：调试模式下使用的样本数量，默认为500
- `--seed`：随机种子，默认为42
- `--print_every`：打印损失的频率，默认为1
- `--plot_every`：记录损失用于绘图的频率，默认为1
- `--gradient_clip`：梯度裁剪阈值，默认为1.0

### 早停参数
- `--patience`：早停耐心值，连续多少个周期损失不下降就停止训练，默认为5
- `--min_delta`：最小损失改善阈值，低于此值视为没有改善，默认为0.001

### 评估参数
- `--eval_only`：仅评估不训练
- `--eval_samples`：评估时使用的样本数量，默认为10

## 使用示例

### 标准训练

```bash
python cmn_eng_translation.py
```

### 调试模式训练

```bash
python cmn_eng_translation.py --debug
```

### 自定义训练参数

```bash
python cmn_eng_translation.py --hidden_size 768 --batch_size 128 --learning_rate 0.0005 --n_epochs 50
```

### 使用早停

```bash
python cmn_eng_translation.py --patience 8 --min_delta 0.0005
```

### 仅评估现有模型

```bash
python cmn_eng_translation.py --eval_only --eval_samples 20
```

### 调整束搜索参数

```bash
python cmn_eng_translation.py --beam_size 5
