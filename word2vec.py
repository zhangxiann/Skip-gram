import collections
import os
import random
import zipfile
import numpy as np
import urllib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

#参数设置
EMBEDDING_DIM = 128 #词向量维度
PRINT_EVERY = 100 #可视化频率
EPOCHES = 1000 #训练的轮数
BATCH_SIZE = 5 #每一批训练数据大小
N_SAMPLES = 3 #负样本大小
WINDOW_SIZE = 5 #周边词窗口大小
FREQ = 5 #词汇出现频数的阈值
DELETE_WORDS = False #是否删除部分高频词
VOCABULARY_SIZE = 50000

url='http://mattmahoney.net/dc/'
def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url+filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise  Exception('Failed to verify '+filename+'. Can you get to it with a browser?')
    return filename

filename=maybe_download('text8.zip', 31344016)

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        # 读取出来的每个单词是 bytes
        data=f.read(f.namelist()[0]).split()
        # 把 bytes 转换为 str
        #data= [str(x, encoding = "utf8") for x in data]
        data = list(map(lambda x: str(x, encoding = "utf8"), data))
    return data

words=read_data(filename)
print('Data size', len(words))

# 取出频数前 50000 的单词

counts_dict = dict((collections.Counter(words).most_common(VOCABULARY_SIZE-1)))
# 去掉频数小于 FREQ 的单词
# trimmed_words = [word for word in words if counts_dict[word] > FREQ]

# 计算 UNK 的频数 = 单词总数 - 前 50000 个单词的频数之和
counts_dict['UNK']=len(words)-np.sum(list(counts_dict.values()))

#建立词和索引的对应
idx_to_word = []
for word in counts_dict.keys():
    idx_to_word.append(word)
word_to_idx = {word:i for i,word in enumerate(idx_to_word)}

#建立词和索引的对应
# idx_to_word = [word for word in counts_dict.keys()]
# word_to_idx = {word:i for i,word in enumerate(idx_to_word)}

# 把单词列表转换为编号的列表
data=list()
for word in words:
    if word in word_to_idx:
        index = word_to_idx[word]
    else:
        index=word_to_idx['UNK']
    data.append(index)

# 把单词列表转换为编号的列表
# data = [word_to_idx.get(word,word_to_idx["UNK"]) for word in words]

# 计算单词频次
total_count = len(data)
word_freqs = {w: c/total_count for w, c in counts_dict.items()}
# 以一定概率去除出现频次高的词汇
if DELETE_WORDS:
    t = 1e-5
    prob_drop = {w: 1-np.sqrt(t/word_freqs[w]) for w in data}
    data = [w for w in data if random.random()<(1-prob_drop[w])]
else:
    data = data

#计算词频,按照原论文转换为3/4次方
word_counts = np.array([count for count in counts_dict.values()],dtype=np.float32)
word_freqs = word_counts/np.sum(word_counts)
word_freqs = word_freqs ** (3./4.)
word_freqs = word_freqs / np.sum(word_freqs)


# DataLoader自动帮忙生成batch
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, data, word_freqs):
        super(WordEmbeddingDataset, self).__init__()
        self.data = torch.Tensor(data).long()  # 解码为词表中的索引
        self.word_freqs = torch.Tensor(word_freqs)  # 词频率

    def __len__(self):
        # 共有多少个item
        return len(self.data)

    def __getitem__(self, idx):
        # 根据idx返回
        center_word = self.data[idx]  # 找到中心词
        pos_indices = list(range(idx - WINDOW_SIZE, idx)) + list(
            range(idx + 1, idx + WINDOW_SIZE + 1))  # 中心词前后各C个词作为正样本
        # pos_indices = list(filter(lambda i: i >= 0 and i < len(self.data), pos_indices))  # 过滤，如果索引超出范围，则丢弃
        pos_indices = [i % len(self.data) for i in pos_indices]
        pos_words = self.data[pos_indices]  # 周围单词
        # 根据 变换后的词频选择 K * 2 * C 个负样本，True 表示可重复采样
        neg_words = torch.multinomial(self.word_freqs, N_SAMPLES * pos_words.shape[0], True)

        return center_word, pos_words, neg_words


# 构造一个神经网络，输入词语，输出词向量
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        # 模型输出nn.Embedding(30000, 100)
        self.out_embed.weight.data.uniform_(-initrange, initrange)  # 权重初始化的一种方法

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        # 模型输入nn.Embedding(30000, 100)
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        # 权重初始化的一种方法

    def forward(self, input_labels, pos_labels, neg_labels):
        # input_labels:[batch_size]
        # pos_labels:[batch_size, windows_size*2]
        # neg_labels:[batch_size, windows_size * N_SAMPLES]

        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, windows_size * 2, embed_size]
        neg_embedding = self.out_embed(neg_labels)  # [batch_size, (windows_size * 2 * K), embed_size]

        # 向量乘法
        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1],新增一个维度用于向量乘法
        # input_embedding = input_embedding.view(BATCH_SIZE, EMBEDDING_DIM, 1)
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze(2)  # [batch_size, windows_size * 2] 只保留前两维
        neg_dot = torch.bmm(neg_embedding.neg(), input_embedding).squeeze(2)  # [batch_size, windows_size * 2 * K] 只保留前两维

        log_pos = F.logsigmoid(pos_dot).sum(1)  # 按照公式计算
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = -(log_pos + log_neg)  # [batch_size]

        return loss

    def input_embeddings(self):
        ##取出self.in_embed数据参数
        return self.in_embed.weight.data.cpu().numpy()


# 构造  dataset 和 dataloader
dataset = WordEmbeddingDataset(data, word_freqs)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 定义一个模型
model = EmbeddingModel(VOCABULARY_SIZE, EMBEDDING_DIM)

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(EPOCHES):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):

        input_labels = input_labels.long()  # 全部转为LongTensor
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()

        optimizer.zero_grad()  # 梯度归零
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("epoch", epoch, "loss", loss.item())

    embedding_weights = model.input_embeddings()
    np.save("embedding-{}".format(EMBEDDING_DIM), embedding_weights)
    torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_DIM))



