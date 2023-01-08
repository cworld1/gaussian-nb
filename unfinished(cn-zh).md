# 使用 Python 模拟朴素贝叶斯算法并求取乳腺癌概率预估值

全概率公式与贝叶斯公式是概率论中重要的公式，主要用于计算比较复杂事件的概率，它们实质上是加法公式和乘法公式的综合运用。概率论与数理统计是研究随机现象统计规律性的一门数学学科，起源于17世纪。发展到现在，已经深入到科学和社会的许多领域。从十七世纪到现在很多国家对这两个公式有了多方面的研究。

概率论的重要课题之一，就是希望从已知的简单事件概率推算出未知的复杂事件的概率。为了达到这个目的，经常把一个复杂的事件分成若干个互不相容事件，再通过分别计算这些简单事件的概率，最后利用概率的可加性得到最终结果。这就是全概率公式的基本思想。把上面的整理清楚就是全概率公式。全概率公式是概率论中一个非常重要的基本公式，通过对概率论课程的研究，发现有多容可以进一步深化与挖掘，从而得到更广泛，更简洁，更实用的结论，以丰富和完善概率论的理论体系。它提供了计算复杂事件概率的一条有效途径，使一个复杂事件的概率计算问题化繁就简。在概率论中起着很重要的作用，灵活使用全概率公式会给我们的解题带来很大方便。蕴涵的数学思想方法：全概率公式蕴含了化整为零，化复杂为简单的数学思想；全概率公式的本质：全概率公式中的 $P(B)$ 是一种平均概率，是条件概率 $P(B|A_i)$ 的加权平均，其中加在每个条件概率上的权重就是作为条件的事件 $A_i$ 发生的概率。

贝叶斯公式首先出现在英国学者 T·贝叶斯（1702-1761) 去世后的1763年的一项著作中。从形式推导上看，这个公式平淡无奇，它不过是条件概率定义与全概率公式的简单推导。其之所以著名，在于其现实乃至哲理意义的解释上：原以为不甚可能的一种情况，可以因某种事件的发生变得甚为可能；或者相反，贝叶斯公式从数量上刻画了这种变化。

## 原理篇

我们用人话而不是大段的数学公式来讲讲朴素贝叶斯是怎么一回事。

###  条件概率

条件概率故名思议就是在一定条件下发生某件事的概率。比如一男子向女生表白成功的概率是20%，记作 $P(A)$ = 20%，其中A代表“该男子向女生表白成功”。而该男子开着捷豹的前提下，向女生表白成功的概率是50%，则记作$P(A|B)$ = 50%，其中B代表“该男子开着捷豹”。毕竟女生都喜欢小动物，像捷豹、路虎、宝马或者悍马什么的。咳咳，跑题了...这个 $P(A|B)$ 就是条件概率了。

### 联合概率

那什么是联合概率呢，该男子开着捷豹且向女生表白成功的概率是1%，则记作 $P(AB)$ = 1%。您可能不禁要问，表白成功的概率不是20%吗？联合概率不是高达50%吗？为什么联合概率这么低？这可能是因为该男子特别穷，开捷豹的概率实在是太低了所以拖累了联合概率。

### 条件概率与联合概率的区别与联系

总结一下，条件概率就是在B的条件下，事件A发生的概率。而联合概率是事件A和B同时发生的概率。二者联系上，可以用如下公式表述：
$$
P(AB) = P(A|B)P(B) = P(B|A)P(A)
$$
即，联合概率等于条件概率乘以条件的概率。
当A和B为独立事件的时候，比如2001年的某一天 Michelle 买了一盘磁带，与此同时中国申奥成功了，这两件事情可以认为是独立事件。
$$
P(AB) = P(A|B)P(B) = P(A)P(B)
$$

### 全概率公式

问题复杂一些，如果事件B不是一个条件，而是一堆条件，这些条件互斥且能穷尽所有可能。比如男生白天向女生表白，黑夜向女生求爱。表白记作事件A，白天记作事件B1，黑夜记作事件B2。不难得到：

$$
P(A) = P(AB1) + P(AB2) = P(B1|A)P(A) + P(B2|A)P(A) = P(A|B1)P(B1) + P(A|B2)P(B2)
$$

其一般形式为：

$$
P(A) = \normalsize\sum_{1}^{m}P(A|B_{i})P(B_{i})
$$

### 贝叶斯公式

算法的名字“朴素贝叶斯”，正是来源于贝叶斯公式。为了便于理解，我们不再使用事件A和事件B而是用机器学习常用的 $x, y$ 来表示：

$$
P(y_{i}|x) = P(x|y_{i})P(y_{i})/P(x)
$$

根据全概率公式，展开分母 $P(A)$，得到：

$$
P(y_{i}|x) = P(x|y_{i})P(y_{i})/\sum_{1}^{m}P(x|y_{j})P(y_{j})
$$

也就是说已知特征x，想要求解 $y_i$，只需要知道先验概率 $P(y_i)$，和似然度 $P(x|y_i)$，即可求解后验概率 $P(y_i|x)$。而对于同一个样本 $x$，$P(x)$ 是一个常量，可以不参与计算。

在全概率公式和贝叶斯公式中，$B_1, B_2, B_3, ..., B_n$ 是伴随结果 $A$ 发生的各种原因，$P(B)$ 是各种原因发生的概率，它一般是有经验给出的，称为先验概率。$P(B_i|A)$ 反映试验后各种情况发生的概率的新结果，可用来修正 $P(B_i)$。“由因索果”用全概率公式，“由果索因”用贝叶斯公式。 

###  高斯分布

如果 $x$ 是连续变量，如何去估计似然度 $P(x|y_i)$ 呢？我们可以假设在 $y_i$ 的条件下，$x$ 服从高斯分布（正态分布）。根据正态分布的概率密度函数即可计算出 $P(x|y_i)$，公式如下：
$$
P(x) = \large\frac{1}{\sigma\sqrt{2\pi}}\normalsize e^{-\frac{(x-\mu)^{2}}{2\sigma^2{}}}
$$

### 高斯朴素贝叶斯

如果 $x$ 是多维的数据，那么我们可以假设 $P(x_1|y_i),P(x_2|y_i)\ ...\ P(x_n|y_i)$ 对应的事件是彼此独立的，这些值连乘在一起得到 $P(x|y_i)$，“彼此独立”也就是朴素贝叶斯的朴素之处。

## 实现篇

### 创建 GaussianNB 类

初始化，存储先验概率、训练集的均值、方差及 label 的类别数量。
```Python
class GaussianNB:
    def __init__(self):
        self.prior = None
        self.avgs = None
        self.vars = None
        self.n_class = None
```

### 计算先验概率

通过 Python 自带的 Counter 计算每个类别的占比，再将结果存储到 numpy 数组中。
```Python
def _get_prior(label: ndarray)->ndarray:
    cnt = Counter(label)
    prior = np.array([cnt[i] / len(label) for i in range(len(cnt))])
    return prior
```

### 计算训练集均值

对每个 label 类别分别计算均值。
```Python
def _get_avgs(self, data: ndarray, label: ndarray)->ndarray:
    return np.array([data[label == i].mean(axis=0) for i in range(self.n_class)])
```

### 计算训练集方差

对每个 label 类别分别计算方差。
```Python
def _get_vars(self, data: ndarray, label: ndarray)->ndarray:
    return np.array([data[label == i].var(axis=0) for i in range(self.n_class)])
```

### 计算似然度

通过高斯分布的概率密度函数计算出似然再连乘得到似然度。
```Python
def _get_likelihood(self, row: ndarray)->ndarray:
    return (1 / sqrt(2 * pi * self.vars) * exp(
        -(row - self.avgs)**2 / (2 * self.vars))).prod(axis=1)
```

### 训练模型

```Python
def fit(self, data: ndarray, label: ndarray):
    self.prior = self._get_prior(label)
    self.n_class = len(self.prior)
    self.avgs = self._get_avgs(data, label)
    self.vars = self._get_vars(data, label)
```

### 预测 prob

用先验概率乘以似然度再归一化得到每个 label 的 prob。
```Python
def predict_prob(self, data: ndarray)->ndarray:
    likelihood = np.apply_along_axis(self._get_likelihood, axis=1, arr=data)
    probs = self.prior * likelihood
    probs_sum = probs.sum(axis=1)
    return probs / probs_sum[:, None]
```

### 预测 label

对于单个样本，取 prob 最大值所对应的类别，也就是 label 的预测值。
```Python
def predict(self, data: ndarray)->ndarray:
    return self.predict_prob(data).argmax(axis=1)
```

## 效果评估

### main 函数

使用著名的乳腺癌数据集，按照 7:3 的比例拆分为训练集和测试集，训练模型，并统计准确度。
```Python
def main():
    print("Tesing the performance of Gaussian NaiveBayes...")
    data, label = load_breast_cancer()
    data_train, data_test, label_train, label_test = train_test_split(data, label, random_state=100)
    clf = GaussianNB()
    clf.fit(data_train, label_train)
    y_hat = clf.predict(data_test)
    acc = _get_acc(label_test, y_hat)
    print("Accuracy is %.3f" % acc)
```

### 效果展示

ACC 0.942，运行时间 22 毫秒。

### 3.3 工具函数

1. run_time - 测试函数运行时间
2. load_breast_cancer - 加载乳腺癌数据
3. train_test_split - 拆分训练集、测试集

## 总结

本文详细介绍了全概率公式、全概率公式的应用、全概率公式的推广及其应用、贝叶斯公式以及它与全概率公式的联系、贝叶斯公式的推广定理及其的应用。通过这些详细的讲述，可以看到两个概率公式的应用是多方面的。两个概率公式的推广将进一步拓展两个概率公式的使用范围，成为我们解决更复杂问题的有效工具。但由于研究周期较短，本研究还有很多不足之处，本文只是举了几个例子来说明它们的应用，事实上它们的应用远不止这一点，还可以用来解决投资、保险、工程等一系列不确定的问题。另外还有什么样的问题应该用全概率公式来解决？什么样的问题应该用贝叶斯公式来解决？什么样的问题要综合两个公式来解决？在什么样的问题中要具体应用几步全概率公式或贝叶斯公式才能解决？本文都没有得出明确的方法和分类，这些都是今后有待进一步深入研究的问题。随着社会的飞速发展，市场竞争日趋激烈，决策者必须综合考察已往的信息及现状从而作出综合判断，决策概率分析这门学科越来越显示其重要性。利用数学方法，定量地对医学问题进行相关分析，使其结论具有可信度，更有利于促进对病人的对症施治等。总之这两个概率公式及推广形式的正确应用有助于进一步研究多个随机过程的试验中目标事件及其条件下各诱发事件的概率，有助于把握随机事件间的相互影响关系，为生产实践提供更有价值的决策信息，进而成为我们解决问题的有效工具。

## 参考