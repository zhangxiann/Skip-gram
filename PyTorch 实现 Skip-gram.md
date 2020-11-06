这是我用于学习 Skip-gram 的笔记。

文中会有一些公式，如果 github 出现公式乱码问题，请通过我的博客查看：[https://zhuanlan.zhihu.com/p/275899732](https://zhuanlan.zhihu.com/p/275899732)。

下面废话不多说，教你手把手实现 Skip-gram。





CBOW 和 Skip-gram 是两种训练得到词向量的方法。其中 CBOW 是从上下文字词推测目标字词，而 Skip-gram 则是从目标字词推测上下文的字词。在大型数据集上，CBOW 比 Skip-gram 效果好；但是在小的数据集上，Skip-gram 比 CBOW 效果好。本文使用 PyTorch 来实现 Skip-gram 模型，主要的论文是：[Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)



以“the quick brown fox jumped over the lazy dog”这句话为例，我们要构造一个上下文单词与目标单词的映射关系，以`quick`为目标单词，假设滑动窗口大小为 1，也就是左边和右边各取 1 个单词作为上下文，这里是`the`和`brown`，可以构造映射关系：(the, quick)，(brown, quick)，这样我们就构造出两个正样本。



此外，对于这个滑动窗口外的其他单词，我们需要构造负样本，但是负样本可以是滑动窗口之外的所有单词。为了减少训练的时间，我们对负样本进行采样 k 个，称为 Negative Sampling。如 k=2，就是对每个正样本，分别构造两个负样本；例如对于`(the, quick)`，采样两个负样本 (lazy , quick)，(dog, quick)。Negative Sampling 的损失函数表示如下：$\underset{\theta}{\arg \max }\log \sigma\left(v_{w_{O}}^{\prime} \top_{w_{I}}\right)+\sum_{i=1}^{k} \mathbb{E}_{w_{i} \sim P_{n}(w)}\left[\log \sigma\left(-v_{w_{i}}^{\prime} T_{w_{I}}\right)\right]$。其中$\sigma(x)$表示 sigmoid 函数，$w_{I}$表示目标单词，$w_{o}$表示正样本的上下文单词，$w_{i}$表示负样本的上下文单词，$v_{w_{O}}^{\prime} \top_{w_{I}}$表示正样本的两个单词向量的内积，我们希望这个内积越大越好，而$v_{w_{i}}^{\prime} T_{w_{I}}$并表示负样本的两个单词向量的内积，我们希望这个内积越小越好，加了负号取反后，也就是希望$-v_{w_{i}}^{\prime} T_{w_{I}}$越大越好。而$\mathbb{E}_{w_{i} \sim P_{n}(w)}$表示从负样本中采样(Negative Sampling)。由于上述损失函数是最大化，取反后变成最小化：$\underset{\theta}{\arg \min }-\log \sigma\left(v_{w_{O}}^{\prime} \top_{w_{I}}\right)-\sum_{i=1}^{k} \mathbb{E}_{w_{i} \sim P_{n}(w)}\left[\log \sigma\left(-v_{w_{i}}^{\prime} T_{w_{I}}\right)\right]$。

我们先导入需要的库

```
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
```

定义全局变量和参数

```
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
```



定义下载文本数据的函数。

Skip-gram 损失函数就是对数损失函数，数据来源于[http://mattmahoney.net/dc/text8.zip](http://mattmahoney.net/dc/text8.zip)，并核对文件大小，如果已经下载了，则跳过。

```
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
```

接下来解压下载的文件，并读取里面的句子每个单词，使用`str()`函数把每个单词从`bytes`转换为`str`。



```
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
```

这里可以使用`for`循环列表表达式来转换：`data= [str(x, encoding = "utf8") for x in data]`，也可以使用`list(map(lambda))`表达式来转换：`data = list(map(lambda x: str(x, encoding = "utf8"), data))`。

如果不了解`map`、`lambda`等用法的同学，请参考[python中lambda,map,filter函数](https://zhuanlan.zhihu.com/p/42091891)。



接下来创建 vocabulary 词汇表，使用`collections.Counter`统计单词列表中单词的频数，然后使用`most_common`方法取出前 50000 个频数最多的单词作为 vocabulary。再创建一个  dict，将单词放入 dict 中，然后把单词转换为编号表示，把 50000 词汇之外的单词设为 UNK(unkown)，编号为 0，并统计 UNK 的数量。有的资料还会去掉低频词，如频次少于 5 的单词会被删除，然后再进行接下来的操作，这里省略这一步。

```
words=read_data(filename)
print('Data size', len(words))

# 取出频数前 50000 的单词

counts_dict = dict((collections.Counter(words).most_common(VOCABULARY_SIZE-1)))
# 去掉频数小于 FREQ 的单词
# trimmed_words = [word for word in words if counts_dict[word] > FREQ]

# 计算 UNK 的频数 = 单词总数 - 前 50000 个单词的频数之和
counts_dict['UNK']=len(words)-np.sum(list(counts_dict.values()))
```

接着建立单词和数字的对应关系，因为我们最终输入到网络的不能是单词的字符串，而是单词对应的编号。

```
#建立词和索引的对应
idx_to_word = []
for word in counts_dict.keys():
	idx_to_word.append(word)
word_to_idx = {word:i for i,word in enumerate(idx_to_word)}	
```

另一种更加简洁的写法是：

```
#建立词和索引的对应
idx_to_word = [word for word in counts_dict.keys()]
word_to_idx = {word:i for i,word in enumerate(idx_to_word)}
```

然后把单词列表转换为编号的列表

```
# 把单词列表转换为编号的列表
data=list()
for word in words:
    if word in word_to_idx:
        index = word_to_idx[word]
    else:
        index=word_to_idx['UNK']
    data.append(index)
```

或者直接使用列表生成式，更加简洁：

```
data = [word_to_idx.get(word,word_to_idx["UNK"]) for word in words]
```

在文本中，如`the`、`a`等词出现频率很高，但是对训练词向量没有太大帮助，为了平衡常见词和少见的词之间的频次，论文中以一定概率丢弃单词，计算公式如下：$P\left(w_{i}\right)=1-\sqrt{\frac{t}{f\left(w_{i}\right)}}$，其中$f(w_{i]})$表示单词的频率，而$t$时超参数，一般$t=10^{-5}$。使用这个公式，那些频率超过$10^{-5}$的单词就会被下采样，同时保持频率大小关系不变。

```
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
```

在负采样时，按照频率来采样单词会导致某些单词次数过多，而少见的单词采样次数过少。论文将词频按照如下公式转换：$P\left(w_{i}\right)=\frac{f\left(w_{i}\right)^{3 / 4}}{\sum_{j=0}^{n} f\left(w_{j}\right)^{3 / 4}}$。按照转换后的词频采样单词，使得最常见的词采样次数减少了，而最少见的词采样次数增加了。下面的代码是计算转换后的词频。

```
#计算词频,按照原论文转换为3/4次方
word_counts = np.array([count for count in counts_dict.values()],dtype=np.float32)
word_freqs = word_counts/np.sum(word_counts)
word_freqs = word_freqs ** (3./4.)
word_freqs = word_freqs / np.sum(word_freqs)
```

下面创建 Dataset，把数据和转换后的词频作为参数传入。在`__getitem__()`中，`idx`是当前的目标单词，根据滑动窗口的大小`WINDOW_SIZE`，取前`WINDOW_SIZE`个词和后`WINDOW_SIZE`个词作为上下文单词。对于每个正样本，采样`N_SAMPLES`个负样本，所以总共采样`N_SAMPLES * pos_words.shape[0]`个负样本。这里根据`word_freqs`词频，使用`torch.multinomial()`来采样。由于`word_freqs`的顺序和`idx_to_word`的顺序是一样的，因此负采样得到索引就是对应单词的编号。

```
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
        pos_indices = list(filter(lambda i: i >= 0 and i < len(self.data), pos_indices))  # 过滤，如果索引超出范围，则丢弃
        pos_words = self.data[pos_indices]  # 周围单词
        # 根据 变换后的词频选择 K * 2 * C 个负样本，True 表示可重复采样
        neg_words = torch.multinomial(self.word_freqs, N_SAMPLES * pos_words.shape[0], True)

        return center_word, pos_words, neg_words

```

这里$WINDOW\_SIZE=5$，$N_SAMPLES=3$，如果正样本的索引没有超过范围，那么会采样 10 个正样本，30 个负样本。

`torch.multinomial()`定义如下：

```
torch.multinomial(input, num_samples,replacement=False, out=None)
```

主要参数：

- input：权重或者采样权重矩阵，元素权重越大，采样概率越大
- num_samples：采样次数
- replacement：是否可以重复采样

input 可以看成一个采样概率矩阵，每一个元素代表其在该行中的采样概率。对 input 的每一行做 n_samples 次取值，输出的张量是 n_samples 次采样得到的元素的索引（下标）。

例子：

```
import torch
weights = torch.Tensor([[0,10,3,0],
					[10,3,0,0],
					[3,0,0,10]])
result = torch.multinomial(weights, 10,replacement=True)
print(result)
```

输出如下：

```
tensor([[1, 1, 1, 2, 2, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 3, 3, 3, 0, 3, 3, 3, 0, 3]])
```

下面定义网络，定义两个`Embedding`层，一个用于映射目标单词，另一个用于映射上下文单词和负样本单词。`Embedding`层的输入维度是 vocabulary_size，输出维度是词向量的长度。`input_embedding`是经过`Embedding`层得到的目标单词向量，形状是`[batch_size,embed_size]`，`pos_embedding`是经过`Embedding`层得到的上下文单词向量，维度是`[batch_size, K,embed_size]`，其中$K$表示上下文单词数量，根据损失函数的定义，要计算目标单词向量和每个上下文单词向量，因此首先把`input_embedding`的形状扩充为`[batch_size,  embed_size, 1]`。对于`pos_embedding`中的每个词向量，分别和`input_embedding`做内积。可以使用`torch.bmm()`方法来实现一个 batch 的向量的相乘。

`torch.bmm()`方法定义如下：

```
torch.bmm(input, mat2, deterministic=False, out=None)
```

参数：

- input：形状是`[batch_size, n,m]`
- mat2：形状是`[batch_size, m,p]`

其中`input`和`mat2`的`batch_size`要相等，对于其中每个元素执行矩阵乘法，$matrix_{[n,m]} \times matrix_{[m,p]}=matrix_{[n,p]}$。

$out_{i}= input _{i} \times {mat} 2_{i}$

最终得到的输出的形状是`[batch_size, n,p]`。

在这里，`input_embedding`的形状扩充为`[batch_size, embed_size, 1]`，`pos_embedding`的形状是`[batch_size, K,embed_size]`，`torch.bmm(pos_embedding, input_embedding)`的结果维度是`[batch_size, K, 1]`。可以实用`unsqueeze()`或者`view()`方法来扩充维度。

`input_embedding`与`neg_embedding`的计算也是同理。代码如下：

```
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

```

定义模型和优化器，开始迭代训练：

```
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

```



有一个问题是：如果在 Dataset 的`__getitem__()`函数中，由于`pos_indices = list(filter(lambda i: i>=0 and i< len(self.data), pos_indices))`可能会造成数据的`size < WINDOW_SIZE*2`，这时可能会导致 DataLoader 无法将一个 Batch 的数据堆叠起来，会报错。有 3种处理方法

1. 可以改为`pos_indices = [i % len(self.data) for i in pos_indices]`，对下表取余，以防超过文档范围
2. 使用自定义`collate_fn`函数来处理这种情况。
3. 或者不使用 PyTorch 的`Dataset `，而是手动生成每个 Batch 的数据。有兴趣的读者可以参考[用Pytorch实现skipgram](https://zhuanlan.zhihu.com/p/82683575)。



**参考**

- [Skip-Gram负采样的Pytorch实现](https://zhuanlan.zhihu.com/p/105955900)
- [用Pytorch实现skipgram](https://zhuanlan.zhihu.com/p/82683575)
- [Tensorflow 实战(黄文坚): Tensorflow 实现 Word2Vec]()
- [python中lambda,map,filter函数](https://zhuanlan.zhihu.com/p/42091891)