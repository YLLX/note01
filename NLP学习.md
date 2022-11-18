# 序列模型

假设$P(x_t|x_{t-1},...,x_1)\approx P(x_t|x_{t-1},...,x_{t-\tau})$，只需要考虑过去的 $\tau$ 个时刻，称为*马尔可夫条件*（Markov condition）。

*自回归模型* （autoregressive modesls）就是对自己执行回归，$ P(x_t|f(x_{t-1},...,x_{t-\tau}))$。根据相关联的变量数的概率公式被称为“一元语法”（unigram）、“二元语法”（bigram）、“三元语法”（trigram）：
$$
P(x_1,x_2,x_3,x_4)=P(x_1,x_2,x_3,x_4)\\
P(x_1,x_2,x_3,x_4)=P(x_1)P(x_2|x_1)P(x_3|x_2)P(x_4|x_3)\\
P(x_1,x_2,x_3,x_4)=p(x_1P(x_2|x_1)P(x_3|x_1,x_2)P(x_4|x_3,x_2)
$$
*隐变量自回归模型*（latent autoregressive models），基于$P(x_t|x_{t-1},...,x_1)=P(x_t|h_t)$来估计$x_t$，其中潜变量$h_t=g(h_{t-1},x_{t-1})$

![自回归](/home/yx/文档/深度学习笔记/imgs/自回归.svg#central)

# 文本预处理

- tokenize，将文本序列拆分为一个个单词/字母
- corpus，语料库，唯一词元的统计，根据出现频率进行排序，将很少出现的词元去掉。另外，还可以加入一些特殊词元，<unk>（未知）、<pad>（填充词）、<bos>（序列开始词）、<eos>（序列结束词）
- vocabulary，将词元映射到数字索引中。

# 循环神经网络（RNN）

![rnn](/home/yx/文档/rnn.svg)
$$
H_t=\Phi(X_tW_{xh}+H_{t-1}W_{hh}+b_h)\\
O_t=H_tW_{hq}+b_q
$$
在NLP中更喜欢使用*困惑度*（perplexity）来衡量模型的质量（n为词元数）：
$$
exp(-\frac{1}{n}\sum_{t=1}^n{logP(x_t|x_{t-1},...,x_1)})
$$

# 门控循环单元（GRU）

[Cho et al., 2014a](https://zh-v2.d2l.ai/chapter_references/zreferences.html#id23)

门控循环单元（gated recurrent unit）由重置门（reset gate）和更新门主成（update gate）。重置门和更新门的计算方式相同，首先进入重置门然后进入更新门。重置门有助于捕获短期依赖关系，将之前的隐变量 $H_{t-1}$有选择的和$X_t$组合，提取短期的依赖关系。更新门有助于捕获长期依赖关系，将$H{}t-1$和$\tilde{H}_t$组合。
$$
R_t=\sigma(X_tW_{xr}+H_{t-1}W_{hr}+b_r)\\
Z_t=\sigma(X_tW_{xz}+H_{t-1}W_{hz}+b_z)\\
\tilde{H}_t=tanh(X_tW_{xh}+(R_t\odot H_{t-1})W_{hh}+b_h)\\
H_t=Z_t\odot H_{t-1}+(1-Z_t)\odot \tilde{H}_t
$$
![gru](/home/yx/文档/gru.svg)

# 长短期记忆神经网络（LSTM）

[Hochreiter & Schmidhuber, 1997](https://zh-v2.d2l.ai/chapter_references/zreferences.html#id68)

长短期记忆网络引入*记忆元*（memory cell）。为了控制记忆元，需要三个门来管理：输入门（input gate）、遗忘门（forget gate）、输出门（output gate）组成。
$$
C_t=F_t\odot C_{t-1}+I_t\odot \tilde{C}_t\\
H_t=O_t\odot tanh(C_t)
$$
![lstm](/home/yx/文档/lstm.svg)

# 深度循环神经网络

![deep-rnn](/home/yx/文档/deep-rnn.svg)

# 双向循环神经网络

[Schuster & Paliwal, 1997](https://zh-v2.d2l.ai/chapter_references/zreferences.html#id146)

[Graves & Schmidhuber, 2005](https://zh-v2.d2l.ai/chapter_references/zreferences.html#id51)

![birnn](/home/yx/文档/birnn.svg)