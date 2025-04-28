from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer
import traceback

# 设置matplotlib中文支持
try:
    from matplotlib.font_manager import FontProperties
    # 尝试设置全局字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 检查是否在Windows系统上运行，添加Windows默认中文字体
    if os.name == 'nt':  # Windows系统
        font_dirs = [os.path.join(os.environ['WINDIR'], 'Fonts')]
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                from matplotlib.font_manager import fontManager
                fontManager.addfont(os.path.join(font_dir, 'simhei.ttf'))
                fontManager.addfont(os.path.join(font_dir, 'simsun.ttc'))
                fontManager.addfont(os.path.join(font_dir, 'msyh.ttc'))
                print("Successfully added Windows Chinese fonts")
except Exception as e:
    print(f"Failed to configure Chinese fonts: {e}")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 特殊标记
SOS_token = 0  # 句子开始标记
EOS_token = 1  # 句子结束标记
PAD_token = 2  # 填充标记

# 全局变量设置
MAX_LENGTH = 15  # 最大句子长度
use_pretrained_tokenizer = True  # 默认使用预训练分词器

# ModernLang类 - 支持预训练分词器
class ModernLang:
    def __init__(self, name, tokenizer=None):
        self.name = name
        self.tokenizer = tokenizer
        if tokenizer:
            self.n_words = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else tokenizer.vocab_size
            self.use_pretrained = True
            
            # 添加index2word属性来满足evaluate()函数的需求
            self.index2word = {}
            # 为特殊标记添加映射
            self.index2word[SOS_token] = "SOS"
            self.index2word[EOS_token] = "EOS"
            self.index2word[PAD_token] = "PAD"
            self.index2word[3] = "<UNK>"
            
            # 从分词器的词表创建反向映射
            self.create_reverse_mapping()
        else:
            # 回退到传统方法
            self.word2index = {}
            self.word2count = {}
            self.index2word = {SOS_token: "SOS", EOS_token: "EOS", PAD_token: "PAD", 3: "<UNK>"}
            self.n_words = 4  # 计数 SOS, EOS, PAD 和 UNK
            self.word2index["<UNK>"] = 3
            self.use_pretrained = False
    
    def create_reverse_mapping(self):
        """为分词器创建索引到单词的映射"""
        if hasattr(self.tokenizer, 'vocab'):
            # 为所有分词器词表项创建反向映射
            for word, idx in self.tokenizer.vocab.items():
                if idx not in self.index2word:  # 不覆盖特殊标记
                    self.index2word[idx] = word
        else:
            # 对于没有vocab属性的分词器，使用get_vocab方法
            try:
                vocab = self.tokenizer.get_vocab()
                for word, idx in vocab.items():
                    if idx not in self.index2word:
                        self.index2word[idx] = word
            except:
                print(f"Warning: Unable to create complete reverse mapping for {self.name} tokenizer")
    
    def tokenize(self, text):
        if self.use_pretrained:
            try:
                # 使用预训练的分词器进行分词
                # 对非常短的句子特殊处理
                if len(text.strip()) <= 3 and text.strip() != "":
                    # 对于非常短的句子，先添加前缀再分词，然后去掉前缀对应的token
                    prefix = "translate: "
                    tokens = self.tokenizer.encode(prefix + text, add_special_tokens=False)
                    # 计算前缀的token数量
                    prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                    # 移除前缀tokens
                    tokens = tokens[len(prefix_tokens):]
                else:
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                
                # 确保tokens不为None
                if tokens is None:
                    print(f"Warning: Tokenizer returned None for '{text}'")
                    tokens = [3]  # 3 is <UNK>
                
                # 确保tokens不包含None值
                if None in tokens:
                    tokens = [3 if t is None else t for t in tokens]
                
                # 如果tokens为空，使用UNK
                if len(tokens) == 0:
                    tokens = [3]  # 3 is <UNK>
                
                return tokens
            except Exception as e:
                # 分词失败时记录错误并返回UNK标记
                print(f"Failed to tokenize text '{text}': {e}")
                return [3]  # 3 is <UNK>
        else:
            # 传统分词
            if self.name == 'eng':
                # 英文按空格分词
                return [self.word2index.get(word, self.word2index.get("<UNK>", 3)) for word in text.split(' ')]
            else:
                # 中文按字符分词
                return [self.word2index.get(char, self.word2index.get("<UNK>", 3)) for char in text]
            
    def decode(self, token_ids):
        if self.use_pretrained:
            return self.tokenizer.decode(token_ids)
        else:
            # 传统解码
            if self.name == 'eng':
                return ' '.join([self.index2word.get(idx, "<UNK>") for idx in token_ids])
            else:
                return ''.join([self.index2word.get(idx, "<UNK>") for idx in token_ids])
    
    def add_sentence(self, sentence):
        if not self.use_pretrained:
            if self.name == 'eng':
                # 英文按空格分词
                for word in sentence.split(' '):
                    self.add_word(word)
            else:
                # 中文按字符分词
                for char in sentence:
                    if char != ' ':  # 忽略空格
                        self.add_word(char)
    
    def add_word(self, word):
        if not self.use_pretrained:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1

# 数据预处理函数
def normalize_english(s):
    """对英文进行标准化处理"""
    # 转为小写，移除非字母字符
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()

def normalize_chinese(s):
    """对中文进行标准化处理"""
    s = s.strip()
    # 在标点符号前添加空格
    s = re.sub(r"([。！？，：；])", r" \1", s)
    return s.strip()

def read_data(data_path):
    """读取数据并分割成中文英文对"""
    print("Reading data...")
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        # 尝试不同的路径
        alternate_paths = [
            'cmn-eng/cmn.txt',
            './cmn-eng/cmn.txt',
            '../cmn-eng/cmn.txt',
            'cmn.txt',
            './cmn.txt'
        ]
        for path in alternate_paths:
            if os.path.exists(path):
                print(f"Found alternative path: {path}")
                data_path = path
                break
        else:
            raise FileNotFoundError(f"Could not find data file: {data_path}")
    
    print(f"Using data file: {data_path}")
    # 读取文件并分割成行
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')
    
    # 分割每行成对，并标准化
    pairs = []
    for l in lines:
        parts = l.strip().split('\t')
        if len(parts) >= 2:
            # 格式为: 英文 \t 中文 \t 版权信息
            eng = normalize_english(parts[0])
            cmn = normalize_chinese(parts[1])
            # 修改为[中文, 英文]的顺序
            pairs.append([cmn, eng])
    
    print(f"Read {len(pairs)} sentence pairs")
    return pairs

def filter_pairs(pairs, max_length=MAX_LENGTH):
    """筛选长度合适的句子对"""
    return [pair for pair in pairs
            if len(pair[1].split(' ')) < max_length and  # 英文单词数
            len(pair[0]) < max_length]  # 中文字符数

def load_tokenizers(pretrained_dir="./pretrained_models"):
    """加载预训练的分词器"""
    try:
        print(f"Loading tokenizers from directory: {pretrained_dir}")
        
        # 检查预训练模型目录是否存在
        if not os.path.exists(pretrained_dir):
            print(f"Warning: Pretrained model directory {pretrained_dir} does not exist")
            raise FileNotFoundError(f"Pretrained model directory {pretrained_dir} does not exist")
        
        # 检查GPT2和BERT目录
        gpt2_path = os.path.join(pretrained_dir, 'gpt2')
        bert_path = os.path.join(pretrained_dir, 'bert-base-chinese')
        
        if not os.path.exists(gpt2_path) or not os.path.exists(bert_path):
            print(f"Warning: Incomplete pretrained model subdirectories, gpt2: {os.path.exists(gpt2_path)}, bert: {os.path.exists(bert_path)}")
        
        print("Loading Chinese tokenizer (BERT)...")
        # 检查必要的BERT文件
        bert_files = {
            'vocab.txt': os.path.exists(os.path.join(bert_path, 'vocab.txt')),
            'tokenizer.json': os.path.exists(os.path.join(bert_path, 'tokenizer.json')),
            'tokenizer_config.json': os.path.exists(os.path.join(bert_path, 'tokenizer_config.json'))
        }
        print(f"BERT file status: {bert_files}")
        
        # 加载BERT分词器
        try:
            # 优先使用from_pretrained方法
            cmn_tokenizer = BertTokenizer.from_pretrained(bert_path)
            print("BERT tokenizer loaded successfully")
        except Exception as e1:
            print(f"Failed to load BERT using from_pretrained method: {e1}")
            try:
                # 尝试直接初始化
                cmn_tokenizer = BertTokenizer(
                    vocab_file=os.path.join(bert_path, 'vocab.txt')
                )
                print("BERT tokenizer loaded successfully via direct initialization")
            except Exception as e2:
                print(f"Failed to initialize BERT tokenizer directly: {e2}")
                # 如果本地加载失败，尝试在线加载
                print("Attempting to load BERT tokenizer online...")
                cmn_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                print("BERT tokenizer loaded from online resources")
        
        print("Loading English tokenizer (GPT2)...")
        # 检查必要的GPT2文件
        gpt2_files = {
            'vocab.json': os.path.exists(os.path.join(gpt2_path, 'vocab.json')),
            'merges.txt': os.path.exists(os.path.join(gpt2_path, 'merges.txt')),
            'tokenizer.json': os.path.exists(os.path.join(gpt2_path, 'tokenizer.json'))
        }
        print(f"GPT2 file status: {gpt2_files}")
        
        # 加载GPT2分词器
        try:
            # 优先使用from_pretrained方法
            eng_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
            print("GPT2 tokenizer loaded successfully")
        except Exception as e1:
            print(f"Failed to load GPT2 using from_pretrained method: {e1}")
            try:
                # 尝试直接初始化
                eng_tokenizer = GPT2Tokenizer(
                    vocab_file=os.path.join(gpt2_path, 'vocab.json'),
                    merges_file=os.path.join(gpt2_path, 'merges.txt')
                )
                print("GPT2 tokenizer loaded successfully via direct initialization")
            except Exception as e2:
                print(f"Failed to initialize GPT2 tokenizer directly: {e2}")
                # 如果本地加载失败，尝试在线加载
                print("Attempting to load GPT2 tokenizer online...")
                eng_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                print("GPT2 tokenizer loaded from online resources")
        
        # 设置特殊标记，确保一致性
        eng_tokenizer.pad_token = eng_tokenizer.eos_token
        
        return cmn_tokenizer, eng_tokenizer  # 注意：交换了返回顺序
    except Exception as e:
        print(f"Failed to load tokenizers: {e}")
        return None, None

def prepare_data(data_path, max_length=MAX_LENGTH, pretrained_dir="./pretrained_models"):
    """准备用于模型训练的数据，包括分词器和句子对"""
    global use_pretrained_tokenizer
    
    # 读取数据
    pairs = read_data(data_path)
    
    # 筛选长度合适的句子对
    pairs = filter_pairs(pairs, max_length)
    print(f"After filtering, {len(pairs)} sentence pairs remain")
    
    # 加载分词器
    if use_pretrained_tokenizer:
        cmn_tokenizer, eng_tokenizer = load_tokenizers(pretrained_dir)  # 注意：交换了顺序
        if cmn_tokenizer and eng_tokenizer:
            input_lang = ModernLang('cmn', cmn_tokenizer)  # 中文作为输入
            output_lang = ModernLang('eng', eng_tokenizer)  # 英文作为输出
            print("Using pretrained tokenizers")
        else:
            print("Pretrained tokenizer loading failed, falling back to traditional tokenization")
            use_pretrained_tokenizer = False
            input_lang = ModernLang('cmn')  # 中文作为输入
            output_lang = ModernLang('eng')  # 英文作为输出
            
            # 构建词表
            print("Building vocabulary...")
            for pair in pairs:
                input_lang.add_sentence(pair[0])  # 中文
                output_lang.add_sentence(pair[1])  # 英文
    else:
        input_lang = ModernLang('cmn')  # 中文作为输入
        output_lang = ModernLang('eng')  # 英文作为输出
        
        # 构建词表
        print("Building vocabulary...")
        for pair in pairs:
            input_lang.add_sentence(pair[0])  # 中文
            output_lang.add_sentence(pair[1])  # 英文
    
    print(f"Input language ({input_lang.name}) vocabulary size: {input_lang.n_words}")
    print(f"Output language ({output_lang.name}) vocabulary size: {output_lang.n_words}")
    
    return input_lang, output_lang, pairs

def split_data(pairs, test_size=0.1, random_state=42):
    """将数据集按照9:1拆分为训练集和测试集"""
    train_pairs, test_pairs = train_test_split(pairs, test_size=test_size, random_state=random_state)
    print(f"Training set size: {len(train_pairs)}, Test set size: {len(test_pairs)}")
    return train_pairs, test_pairs

def indexes_from_sentence(lang, sentence):
    """从句子获取索引列表"""
    try:
        if lang.use_pretrained:
            # 使用预训练分词器
            tokens = lang.tokenize(sentence)
            # 确保没有None值
            if None in tokens:
                print(f"Warning: Sentence '{sentence}' tokenization resulted in None values")
                tokens = [3 if t is None else t for t in tokens]  # 3 is <UNK>
            # 确保返回的是列表，不是None
            if tokens is None:
                tokens = [3]  # <UNK>
            return [t for t in tokens if t is not None]
        else:
            # 传统分词
            if lang.name == 'cmn':
                # 中文按字符分词
                indexes = [lang.word2index.get(char, 3) for char in sentence]  # 3 is <UNK>
            else:
                # 英文按空格分词
                indexes = [lang.word2index.get(word, 3) for word in sentence.split(' ')]
            return indexes
    except Exception as e:
        print(f"Error processing sentence '{sentence}': {e}")
        # 返回一个简单的UNK标记作为回退
        return [3]  # 3 is <UNK>

def get_dataloader(pairs, input_lang, output_lang, batch_size=32, shuffle=True):
    """创建PyTorch数据加载器"""
    n = len(pairs)
    print(f"Creating dataloader for {n} pairs with batch size {batch_size}")
    
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    
    for idx, (inp, tgt) in enumerate(pairs):
        # 每1000个样本打印一次进度
        if idx % 1000 == 0 or idx == n-1:
            print(f"Processing pair {idx+1}/{n}")
            
        try:
            # 获取句子的索引表示
            inp_ids = indexes_from_sentence(input_lang, inp)
            tgt_ids = indexes_from_sentence(output_lang, tgt)
            
            # 确保索引是有效的整数列表
            if not inp_ids:  # 如果为空
                print(f"Warning: Input sentence '{inp}' resulted in empty index list, using <UNK>")
                inp_ids = [3]  # 3 is <UNK>
                
            if not tgt_ids:  # 如果为空
                print(f"Warning: Target sentence '{tgt}' resulted in empty index list, using <UNK>")
                tgt_ids = [3]  # 3 is <UNK>
            
            # 添加EOS标记
            inp_ids.append(EOS_token)
            tgt_ids.append(EOS_token)
            
            # 填充到最大长度
            if len(inp_ids) > MAX_LENGTH:
                inp_ids = inp_ids[:MAX_LENGTH]
            if len(tgt_ids) > MAX_LENGTH:
                tgt_ids = tgt_ids[:MAX_LENGTH]
                
            # 确保所有ID都是有效整数
            input_ids[idx, :len(inp_ids)] = np.array(inp_ids, dtype=np.int32)
            target_ids[idx, :len(tgt_ids)] = np.array(tgt_ids, dtype=np.int32)
            
        except Exception as e:
            print(f"Error processing sentence pair [{inp} -> {tgt}]: {e}")
            # 使用UNK填充
            input_ids[idx, 0] = 3  # <UNK>
            target_ids[idx, 0] = 3  # <UNK>
            input_ids[idx, 1] = 1  # EOS
            target_ids[idx, 1] = 1  # EOS
    
    # 创建PyTorch数据集
    data = TensorDataset(torch.LongTensor(input_ids).to(device),
                         torch.LongTensor(target_ids).to(device))
    
    # 创建数据加载器，确保drop_last=False以便处理所有样本
    if shuffle:
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last=False)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, drop_last=False)
    
    # 验证数据加载器大小
    expected_batches = (n + batch_size - 1) // batch_size  # 向上取整
    actual_batches = len(dataloader)
    if expected_batches != actual_batches:
        print(f"WARNING: Expected {expected_batches} batches but got {actual_batches} batches")
    else:
        print(f"Created dataloader with {actual_batches} batches")
    
    return dataloader

# 编码器模型
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

# Bahdanau注意力机制
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
        
    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        
        return context, weights

# 带注意力机制的解码器
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        max_length = MAX_LENGTH
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []
        
        # 如果目标张量存在，使用其长度，否则使用最大长度
        target_length = target_tensor.size(1) if target_tensor is not None else max_length
        
        use_teacher_forcing = True if target_tensor is not None else False
        
        for i in range(target_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)
            
            if use_teacher_forcing and i < target_length - 1:
                # 教师强制：使用目标作为下一个输入
                decoder_input = target_tensor[:, i:i+1]
            else:
                # 无教师强制：使用自己的预测作为下一个输入
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # 从历史中分离
                
                # 如果所有批次都预测到了EOS，可以提前结束
                if (decoder_input == EOS_token).all():
                    break
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        
        # 确保注意力张量维度正确
        if attentions:
            attentions = torch.cat(attentions, dim=1)
        else:
            attentions = torch.zeros(batch_size, 1, encoder_outputs.size(1), device=device)
        
        return decoder_outputs, decoder_hidden, attentions
    
    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        
        return output, hidden, attn_weights

# 训练函数
def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    batch_count = 0
    total_samples = 0
    
    encoder.train()
    decoder.train()
    
    # 计算总样本数和预期batch数
    expected_batches = len(dataloader)
    print(f"Expected batches: {expected_batches}, Batch size: {dataloader.batch_size}")
    
    for data in dataloader:
        batch_count += 1
        input_tensor, target_tensor = data
        total_samples += input_tensor.size(0)  # 累计处理的样本数
        
        # 输出当前处理的批次信息
        if batch_count == 1 or batch_count % 50 == 0 or batch_count == expected_batches:
            print(f"Processing batch {batch_count}/{expected_batches}, samples so far: {total_samples}")
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch completed: processed {batch_count} batches, {total_samples} samples in total")
    
    return total_loss / max(1, batch_count)  # 避免除以零的情况

# 时间显示辅助函数
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s:.0f}s'

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return f'{as_minutes(s)} (remaining: {as_minutes(rs)})'

# 完整训练流程
def train(train_dataloader, encoder, decoder, n_epochs=20, learning_rate=0.001, print_every=1, plot_every=1, 
          encoder_optimizer=None, decoder_optimizer=None, criterion=None, patience=5, min_delta=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    
    # 如果未提供优化器和损失函数，则创建默认的
    if encoder_optimizer is None:
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    if decoder_optimizer is None:
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    if criterion is None:
        criterion = nn.NLLLoss(ignore_index=PAD_token)
    
    # 早停相关变量
    best_loss = float('inf')
    patience_counter = 0
    early_stop = False
    
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f'{time_since(start, epoch/n_epochs)} ({epoch}/{n_epochs} {epoch/n_epochs*100:.1f}%) Loss: {print_loss_avg:.4f}')
        
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            
            # 早停检查
            if plot_loss_avg < best_loss - min_delta:
                best_loss = plot_loss_avg
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered: Loss did not improve for {patience} epochs")
                    early_stop = True
                    break
    
    if early_stop:
        print(f"Training stopped at epoch {epoch}, best loss: {best_loss:.4f}")
    
    return plot_losses

# 绘制损失曲线
def show_plot(points):
    plt.figure()
    plt.plot(points)
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig('training_loss.png')
    plt.show()

# 评估函数
def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        # 将模型设为评估模式
        encoder.eval()
        decoder.eval()
        
        # 输出原始输入句子
        print(f"Original input: '{sentence}'")
        
        # 处理输入句子
        indexes = indexes_from_sentence(input_lang, sentence)
        
        # 确保索引不为空
        if not indexes:
            print(f"Warning: Sentence '{sentence}' resulted in empty tokenization, using <UNK> instead")
            indexes = [3]  # <UNK>
            
        indexes.append(EOS_token)
        
        # 确保长度不超过最大值
        if len(indexes) > MAX_LENGTH:
            indexes = indexes[:MAX_LENGTH]
            
        input_tensor = torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)
        
        # 编码
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        
        # 解码 - 使用beam search以提高译文质量
        decoded_words, attentions = beam_search_decode(decoder, encoder_outputs, encoder_hidden, output_lang, beam_size=3)
        
        # 调试信息
        print(f"Input indexes: {indexes}")
        print(f"Output words: {decoded_words}")
        
        # 如果输出是字符串，转换为字符串数组以便绘制注意力
        if isinstance(decoded_words, str):
            decoded_words = decoded_words.split()
        
        return decoded_words, attentions

# 添加Beam Search解码函数
def beam_search_decode(decoder, encoder_outputs, encoder_hidden, output_lang, beam_size=3, max_length=MAX_LENGTH):
    """使用束搜索解码，提高翻译质量"""
    with torch.no_grad():
        batch_size = encoder_outputs.size(0)
        
        # 初始化第一个输入为SOS标记
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        
        # 初始束为一个元素：(累积概率(log prob), 序列, 隐藏状态, 注意力)
        beams = [(0.0, [SOS_token], decoder_hidden, [])]
        completed_beams = []
        
        # 束搜索过程
        for _ in range(max_length):
            candidates = []
            for cumulative_prob, seq, hidden, attns in beams:
                # 如果序列已经结束，添加到已完成的束中
                if seq[-1] == EOS_token:
                    completed_beams.append((cumulative_prob, seq, attns))
                    continue
                    
                # 设置解码器输入
                decoder_input = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
                
                # 计算下一步
                decoder_output, decoder_hidden, attn_weights = decoder.forward_step(
                    decoder_input, hidden, encoder_outputs
                )
                
                # 获取topk预测
                log_probs = F.log_softmax(decoder_output, dim=-1)
                topk_probs, topk_ids = log_probs.topk(beam_size)
                
                # 为每个topk创建新的候选
                for i in range(beam_size):
                    new_prob = cumulative_prob + topk_probs[0, 0, i].item()
                    new_seq = seq + [topk_ids[0, 0, i].item()]
                    new_attns = attns + [attn_weights]
                    candidates.append((new_prob, new_seq, decoder_hidden, new_attns))
            
            # 如果所有序列都结束，则退出循环
            if len(completed_beams) >= beam_size:
                break
                
            # 选择最佳的beam_size候选
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
        
        # 如果没有完成的序列，使用当前最佳的序列
        if not completed_beams and beams:
            best_prob, best_seq, best_attns = beams[0][0], beams[0][1], beams[0][3]
        else:
            # 选择已完成序列中得分最高的
            completed_beams = sorted(completed_beams, key=lambda x: x[0], reverse=True)
            if completed_beams:
                best_prob, best_seq, best_attns = completed_beams[0]
            else:
                # 如果没有完成的序列，返回UNK
                return ["<UNK>"], torch.zeros(1, 1, 1, device=device)
        
        # 调试：显示选择的最佳序列
        print(f"Best sequence (token IDs): {best_seq}")
        
        # 移除SOS和EOS标记
        words = []
        for idx in best_seq[1:]:
            if idx == EOS_token:
                break
            if idx == PAD_token:
                continue
            
            # 使用输出语言的分词器或词表
            words.append(idx)
        
        # 如果没有生成任何有效单词，添加UNK
        if not words:
            words = [3]  # 3代表<UNK>
            
        print(f"Processed token IDs: {words}")
            
        # 解码生成的单词ID序列为文本
        if output_lang.use_pretrained:
            # 使用预训练分词器解码整个序列
            try:
                # 确保所有token都在词表中
                valid_tokens = [token for token in words if token < output_lang.tokenizer.vocab_size]
                if not valid_tokens:
                    print("Warning: No valid tokens for decoder")
                    valid_tokens = [3]  # <UNK>
                
                # 使用GPT2分词器直接解码
                decoded_text = output_lang.tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
                print(f"Decoded text: '{decoded_text}'")
                
                # 如果解码结果为空，回退到单个token解码
                if not decoded_text:
                    print("Warning: Empty decoded text, falling back to per-token decoding")
                    decoded_words = []
                    for token_id in valid_tokens:
                        token_text = output_lang.tokenizer.decode([token_id], skip_special_tokens=True).strip()
                        if token_text:
                            decoded_words.append(token_text)
                        else:
                            decoded_words.append("<UNK>")
                    return decoded_words if decoded_words else ["<UNK>"], torch.cat(best_attns, dim=1).cpu() if best_attns else torch.zeros(1, 1, encoder_outputs.size(1), device='cpu')
                
                # 返回以空格分隔的词列表
                words_list = decoded_text.split()
                return words_list if words_list else ["<UNK>"], torch.cat(best_attns, dim=1).cpu() if best_attns else torch.zeros(1, 1, encoder_outputs.size(1), device='cpu')
                
            except Exception as e:
                print(f"Failed to decode using pretrained tokenizer: {e}")
                # 尝试一个一个解码
                decoded_words = []
                for word_id in words:
                    try:
                        if word_id < output_lang.tokenizer.vocab_size:
                            word = output_lang.tokenizer.decode([word_id], skip_special_tokens=True).strip()
                            if word:
                                decoded_words.append(word)
                            else:
                                decoded_words.append("<UNK>")
                        else:
                            decoded_words.append("<UNK>")
                    except Exception as e2:
                        print(f"Error decoding token {word_id}: {e2}")
                        decoded_words.append("<UNK>")
                return decoded_words if decoded_words else ["<UNK>"], torch.cat(best_attns, dim=1).cpu() if best_attns else torch.zeros(1, 1, encoder_outputs.size(1), device='cpu')
        else:
            # 使用传统词表解码
            decoded_words = []
            for idx in words:
                if idx in output_lang.index2word:
                    decoded_words.append(output_lang.index2word[idx])
                else:
                    decoded_words.append("<UNK>")
            
            return decoded_words, torch.cat(best_attns, dim=1).cpu() if best_attns else torch.zeros(1, 1, encoder_outputs.size(1), device='cpu')

# 随机评估一些样本
def evaluate_randomly(encoder, decoder, pairs, input_lang, output_lang, n=10):
    results = []
    
    for i in range(min(n, len(pairs))):
        try:
            pair = random.choice(pairs)
            print(f'Input: {pair[0]}')
            print(f'Target: {pair[1]}')
            
            try:
                output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
                
                # 构建输出句子
                if output_lang.use_pretrained:
                    if isinstance(output_words, list):
                        # 如果已经是单词列表，直接连接
                        output_sentence = ' '.join(output_words)
                    else:
                        # 字符串情况
                        output_sentence = output_words
                else:
                    # 传统解码，已经是单词列表
                    output_sentence = ' '.join(output_words)
                
                print(f'Prediction: {output_sentence}')
                print('')
                
                # 可视化注意力权重
                if i == 0 and attentions is not None and attentions.shape[0] > 0:
                    try:
                        show_attention(pair[0], output_words, attentions)
                    except Exception as e:
                        print(f"Attention visualization failed: {e}")
                
                results.append({
                    'Input': pair[0],
                    'Target': pair[1],
                    'Prediction': output_sentence
                })
            except Exception as e:
                print(f"Error evaluating sample [{pair[0]} -> {pair[1]}]: {e}")
                traceback_info = traceback.format_exc()
                print(f"Detailed error: {traceback_info}")
                results.append({
                    'Input': pair[0],
                    'Target': pair[1],
                    'Prediction': '[Error: Evaluation failed]'
                })
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
    
    try:
        # 创建DataFrame保存结果
        results_df = pd.DataFrame(results)
        results_df.to_csv('translation_results.csv', index=False)
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
    
    return results

# 可视化注意力权重
def show_attention(input_sentence, output_words, attentions):
    """可视化注意力权重，使用替代方法显示中文字符"""
    # 确保注意力张量维度正确
    if attentions is None or attentions.size == 0:
        print("Warning: Attention tensor is empty")
        return
        
    # 将注意力转为numpy数组，确保为2D
    if isinstance(attentions, torch.Tensor):
        attention_np = attentions.numpy()
    else:
        attention_np = attentions
        
    # 如果是3D或更高维度，取第一个样本
    if len(attention_np.shape) > 2:
        attention_np = attention_np[0]
    
    # 改进中文输入处理 - 按字符分割而非按空格
    if any(ord(c) > 127 for c in input_sentence):  # 检测是否包含中文/非ASCII字符
        # 中文处理：直接按字符分割
        input_words = list(input_sentence.replace(' ', ''))
        # 创建替代标签 (使用序号代替中文字符)
        input_labels = [f"char{i+1}" for i in range(len(input_words))]
        # 打印中文字符和它们的替代标签对应关系
        print("Chinese characters mapping:")
        for i, (char, label) in enumerate(zip(input_words, input_labels)):
            print(f"{label}: {char}")
    else:
        # 英文处理：按空格分割
        input_words = input_sentence.split()
        input_labels = input_words
        
    # 确保注意力矩阵正确切片
    valid_len = min(len(input_words), attention_np.shape[1])
    if valid_len == 0:
        print("Warning: Input sentence has 0 length after processing")
        return
        
    # 对注意力矩阵进行合理切片
    att_matrix = attention_np[:, :valid_len]
    
    # 处理输出词
    if len(output_words) == 0:
        output_words = ["<UNK>"]
    
    # 如果输出是字符串，转换为列表
    if isinstance(output_words, str):
        output_words = output_words.split()
    
    # 确保输入输出长度与注意力矩阵维度一致
    valid_output_len = min(len(output_words), att_matrix.shape[0])
    valid_input_len = min(len(input_words), att_matrix.shape[1])
    
    if valid_output_len == 0 or valid_input_len == 0:
        print("Warning: Valid input or output length is 0")
        return
    
    # 打印调试信息
    print(f"Input words: {input_words[:valid_input_len]}")
    print(f"Output words: {output_words[:valid_output_len]}")
    print(f"Attention matrix shape: {att_matrix.shape}")
    
    # 绘制热图，不使用中文标签
    plt.figure(figsize=(12, 8))
    
    # 创建热图
    plt.imshow(att_matrix[:valid_output_len, :valid_input_len], cmap='viridis')
    plt.colorbar()
    
    # 设置替代标签
    plt.xticks(range(valid_input_len), input_labels[:valid_input_len], rotation=45)
    plt.yticks(range(valid_output_len), output_words[:valid_output_len])
    
    # 添加值标签
    for i in range(valid_output_len):
        for j in range(valid_input_len):
            weight = att_matrix[i, j]
            plt.text(j, i, f'{weight:.2f}', 
                     ha="center", va="center", color="white" if weight > 0.2 else "black")
    
    # 添加标题和说明
    plt.title("Attention Weights")
    plt.xlabel("Input (source) tokens")
    plt.ylabel("Output (target) tokens")
    plt.tight_layout()
    
    # 添加额外的图例说明中文字符映射
    if any(ord(c) > 127 for c in input_sentence):
        legend_text = "Chinese characters:\n"
        for i, (char, label) in enumerate(zip(input_words[:valid_input_len], input_labels[:valid_input_len])):
            legend_text += f"{label}: {char}\n"
        plt.figtext(0.01, 0.01, legend_text, fontsize=9, wrap=True)
    
    # 保存并显示
    plt.savefig('attention_weights.png', dpi=150, bbox_inches='tight')
    plt.show()

# 主函数
def main(debug=False, data_path=None, pretrained_dir='./pretrained_models', model_path='translation_model.pt',
         hidden_size=None, dropout=0.1, max_length=15, n_epochs=None, batch_size=None, learning_rate=0.001,
         beam_size=3, sample_size=1000, seed=42, print_every=1, plot_every=1, gradient_clip=1.0,
         eval_only=False, eval_samples=10, patience=5, min_delta=0.001):
    # 设置随机种子
    random.seed(seed)
    torch.manual_seed(seed)
    
    # 数据准备
    if data_path is None:
        data_path = 'cmn-eng/cmn.txt'  # 默认路径
    
    print(f"Using pretrained model directory: {pretrained_dir}")
    input_lang, output_lang, pairs = prepare_data(data_path, max_length=max_length, pretrained_dir=pretrained_dir)
    
    print(f"Total available sentence pairs: {len(pairs)}")
    
    # 在调试模式下，仅使用少量数据
    if debug:
        print(f"Debug mode: Using limited data ({sample_size} samples) for quick testing")
        pairs = pairs[:sample_size]  # 使用sample_size个句子对以更好地训练模型
        print(f"Selected {len(pairs)} pairs for debug mode")
    else:
        print(f"Using all {len(pairs)} pairs for training/testing")
    
    # 划分训练集和测试集
    train_pairs, test_pairs = split_data(pairs)
    
    # 创建数据加载器
    if batch_size is None:
        batch_size = 32 if debug else 64  # 默认batch size
    
    print(f"Using batch size: {batch_size}")
    print(f"Creating data loader for {len(train_pairs)} training pairs...")
    
    train_dataloader = get_dataloader(train_pairs, input_lang, output_lang, batch_size=batch_size)
    
    print(f"Data loader created with {len(train_dataloader)} batches")
    
    # 初始化模型
    if hidden_size is None:
        hidden_size = 256 if debug else 512  # 增大隐藏层大小以提高模型容量
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
    
    # 设置优化器和损失函数
    if n_epochs is None:
        n_epochs = 30 if debug else 50  # 增加训练轮次以获得更好的性能
    if learning_rate is None:
        learning_rate = 0.001  # 学习率
    print(f"Training settings: epochs={n_epochs}, learning_rate={learning_rate}, batch_size={batch_size}, hidden_size={hidden_size}")
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_token)
    
    # 训练模型
    try:
        plot_losses = train(
            train_dataloader, 
            encoder, 
            decoder, 
            n_epochs=n_epochs, 
            learning_rate=learning_rate,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            criterion=criterion,
            print_every=print_every,
            plot_every=plot_every,
            patience=patience,
            min_delta=min_delta
        )
        
        # 绘制损失曲线
        try:
            show_plot(plot_losses)
        except Exception as e:
            print(f"Error plotting loss curve: {e}")
            
        # 保存模型
        try:
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'input_lang': input_lang,
                'output_lang': output_lang
            }, model_path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")
    except Exception as e:
        print(f"Error training model: {e}")
    
    # 设置为评估模式
    encoder.eval()
    decoder.eval()
    
    # 在测试集上评估
    if eval_only:
        n_examples = eval_samples  # 调试模式评估更少的例子
        results = evaluate_randomly(encoder, decoder, test_pairs, input_lang, output_lang, n=n_examples)
    else:
        n_examples = 3 if debug else 10  # 调试模式评估更少的例子
        results = evaluate_randomly(encoder, decoder, test_pairs, input_lang, output_lang, n=n_examples)
    
    # 可视化注意力权重（以一个样本为例）
    sample_pair = test_pairs[0]
    output_words, attentions = evaluate(encoder, decoder, sample_pair[0], input_lang, output_lang)
    show_attention(sample_pair[0], output_words, attentions)
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Chinese to English Neural Machine Translation')
    parser.add_argument('--debug', action='store_true', help='Debug mode: use less data and epochs for quick testing')
    parser.add_argument('--data_path', type=str, default='cmn-eng/cmn.txt', help='Data file path')
    parser.add_argument('--pretrained_dir', type=str, default='./pretrained_models', help='Pretrained models directory')
    parser.add_argument('--model_path', type=str, default='translation_model.pt', help='Model save/load path')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size, 256 for debug mode, 512 for normal')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_length', type=int, default=15, help='Maximum sentence length')
    
    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of training epochs, 30 for debug mode, 50 for normal')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size, 32 for debug mode, 64 for normal')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beam_size', type=int, default=3, help='Beam search size')
    parser.add_argument('--sample_size', type=int, default=1000, help='Number of samples to use in debug mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--print_every', type=int, default=1, help='Print loss frequency (every N epochs)')
    parser.add_argument('--plot_every', type=int, default=1, help='Record loss for plotting (every N epochs)')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping threshold')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience: how many epochs to wait for improvement')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum improvement threshold for early stopping')
    
    # Evaluation parameters
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate without training')
    parser.add_argument('--eval_samples', type=int, default=10, help='Number of samples to evaluate')
    
    args = parser.parse_args()
    
    # Pass command line arguments to main function
    main(
        debug=args.debug, 
        data_path=args.data_path,
        pretrained_dir=args.pretrained_dir,
        model_path=args.model_path,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        max_length=args.max_length,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beam_size=args.beam_size,
        sample_size=args.sample_size,
        seed=args.seed,
        print_every=args.print_every,
        plot_every=args.plot_every,
        gradient_clip=args.gradient_clip,
        eval_only=args.eval_only,
        eval_samples=args.eval_samples,
        patience=args.patience,
        min_delta=args.min_delta
    ) 