# KWS论文阅读笔记
> 为撰写综述做准备，内容来自2021-2023的ICASSP, SE, TASL
***
## 1. A LIGHTWEIGHT DYNAMIC FILTER FOR KEYWORD SPOTTING
> Kim D, Ko K, Kwak J, et al. A Lightweight Dynamic Filter For Keyword Spotting[C]//2023 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW). IEEE, 2023: 1-5.  (韩国高丽大学，美国德雷塞尔大学) (这篇基本没看懂，不太明白像素级实例级动态滤波之类的)
### Abstract
Keyword Spotting (KWS) from speech signals is widely applied toperform fully hands-free speech recognition. The KWS network isdesigned as a small-footprint model so it can continuously be active.Recent efforts have explored dynamic filter-based models in deeplearning frameworks to enhance the system’s robustness or accuracy.However, as a dynamic filter framework requires high computational costs, the implementation is limited to the computational conditionof the device. In this paper, we propose a lightweight dynamic filterto improve the performance of KWS. Our proposed model dividesthe dynamic filter into two branches to reduce computational complexity: pixel level and instance level. The proposed lightweight dynamic filter is applied to the front end of KWS to enhance the separability of the input data. The experimental results show that our model is robustly working on unseen noise and small training data environments by using a small computational resource. 

*Index Terms: keyword spotting, dynamic filter, dynamic weight,computational cost*  

（语音信号关键字识别(KWS)被广泛应用于全免提语音识别。KWS网络被设计成一个小占用模型，因此它可以持续活动。最近的研究探索了深度学习框架中基于动态过滤器的模型，以提高系统的鲁棒性或准确性。然而，由于动态滤波器框架需要较高的计算成本，其实现受到设备计算条件的限制。在本文中，我们提出了一种轻量级的动态滤波器来提高KWS的性能。我们提出的模型将动态滤波分为两个分支:像素级和实例级，以降低计算复杂度。将所提出的轻量级动态滤波器应用于KWS前端，增强了输入数据的可分性。实验结果表明，该模型可以在不可见噪声和小型训练数据环境下稳健地工作）
### Method  
![](img/mk-2023-09-29-10-53-13.png)  
其实细节还不是特别懂，写综述的时候对照论文写吧
### Experiment and Result  
#### Exprtimental Setup  
- Dataset: [google speech command datasets v1 and v2](https://arxiv.org/abs/1804.03209)  
  we utilized 10 keywords with two extra classes (unknown or silent) for model training, injected background noise, and added random time-shifting.
- Noise Dataset: [DCASE](https://archive.nyu.edu/handle/2451/60751), [Urbansound8K](https://dl.acm.org/doi/abs/10.1145/2647868.2655045) and [WHAM](https://arxiv.org/abs/1907.01160) 
- Feature: MFCC, 30ms of windows with 10ms overlap, 16KHz,  40 MFCC coefficients, [40,98] (98怎么来的?)  
- BatchSize: 100, 30K iterations, Adam optimizer with a 0.001 initial learning rate, every 10K iteration, the learning rate is decreased by 0.1.  
- In the PDF and the dynamic convolution process, we used 3 × 3 CNN kernel (k = 3) dilated by (2,2) with a stride of 1. In the IDF, the first FC and second FC follow 40 × 40 and 40×k layer dimensions respectively. For DIN, two layers of FC which have 40 × 40 filter size respectively are utilized to produce α and β  
#### Result
![](img/mk-2023-09-29-12-59-36.png)  
![](img/mk-2023-09-29-12-59-57.png)

### 知识补充
> 补充完相关知识后懂了一点  

[动态滤波器（动态卷积网络）](http://t.csdnimg.cn/4mE4E)  
[Dynamic Convolution: Attention over Convolution Kernels](https://arxiv.org/abs/1912.03458)  
- 动态感知器  
  ![](img/mk-2023-09-29-11-24-45.png)
- 动态卷积  
  ![](img/mk-2023-09-29-11-25-58.png) 


  ***
## 2. A Novel Loss Function and Training Strategy for Noise-Robust Keyword Spotting  
> López-Espejo I, Tan Z H, Jensen J. A novel loss function and training strategy for noise-robust keyword spotting[J]. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2021, 29: 2254-2266.(丹麦奥尔堡大学)  

### Abstract  
The development of keyword spotting (KWS) systems that are accurate in noisy conditions remains a challenge. Towards this goal, in this paper we propose a novel training strategy relying on multi-condition training for noise-robust KWS. By this strategy,
we think of the state-of-the-art KWS models as the composition of a keyword embedding extractor and a linear classifier that are successively trained. To train the keyword embedding extractor, we also propose a new (CN,2 + 1)-pair loss function extending
the concept behind related loss functions like triplet and N-pair losses to reach larger inter-class and smaller intra-class variation. Experimental results on a noisy version of the Google Speech Commands Dataset show that our proposal achieves around 12% KWS accuracy relative improvement with respect to standard end-to-end multi-condition training when speech is distorted by unseen noises. This performance improvement is achieved without increasing the computational complexity of the KWS model.  

*Index Terms: Keyword spotting, noise robustness, multicondition training, deep metric learning, loss function, keyword embedding.*  

（开发在噪声条件下准确的关键字定位(KWS)系统仍然是一个挑战。为此，本文提出了一种基于多条件训练的噪声鲁棒KWS训练策略。通过这种策略，我们认为最先进的KWS模型是一个关键字嵌入提取器和一个连续训练的线性分类器的组合。为了训练关键字嵌入提取器，我们还提出了一个新的(CN,2 + 1)对损失函数，扩展了相关损失函数(如三元组和n对损失)背后的概念，以达到更大的类间和更小的类内变化。在谷歌语音命令数据集的噪声版本上的实验结果表明，当语音被看不见的噪声扭曲时，我们的建议相对于标准的端到端多条件训练实现了大约12%的KWS精度提高。在不增加KWS模型的计算复杂性的情况下实现了这种性能改进。）  

### Keyword Spotting Training Strategy  
简单来说就是把大家提出来的各种KWS模型分两部分来训练，一部分是特征提取部分（整个模型去掉最后带softmax的线性分类层），这部分用(CN,2 + 1)对损失函数训练，另一部分是带softmax的线性分类层，用交叉熵损失训练。  

- Training of the keyword embedding extractor: First, the keyword embedding extractor is multi-condition trained by considering a new (CN,2 + 1)-pair loss function。  
- Training of the linear classifier: Second, the linear classifier is trained using multi-condition keyword embeddings and cross-entropy loss.  
###  A New (CN,2 + 1)-Pair Loss Function  
![](img/mk-2023-09-29-14-06-34.png)  
比较底层，艰深， 晦涩，暂时不花时间。。。  
### Experiment and Result  
#### Exprtimental Setup  
- Dataset: [google speech command datasets](https://arxiv.org/abs/1804.03209) (进行了一些加噪声的处理)
- Noise Dataset:  [NOISEX-92](https://www.sciencedirect.com/science/article/abs/pii/0167639393900953), [CHiME-3](https://www.sciencedirect.com/science/article/pii/S088523081630122X)   
- Model: [来自Icassp2018](https://ieeexplore.ieee.org/abstract/document/8462688)  
#### Result
the best performance for unseen noises is clearly obtained by the proposed method (∼83.53% acc.) in a statistically significant manner. In particular, on average, our training strategy using the (CN,2 + 1)-pair loss function yields around 12% KWS accuracy relative improvement with respect to Baseline.  


***
## 3. Autokws: Keyword spotting with differentiable architecture search  
> Zhang B, Li W, Li Q, et al. Autokws: Keyword spotting with differentiable architecture search[C]//ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021: 2830-2834.(小米AI实验室)  

### Abstract  
Smart audio devices are gated by an always-on lightweight keyword spotting program to reduce power consumption. It is however challenging to design models that have both high accuracy and low latency for accurate and fast responsiveness. Many efforts have been made to develop end-to-end neural networks, in which depthwise separable convolutions, temporal convolutions, and LSTMs are adopted as building units. Nonetheless, these networks designed with human expertise may not achieve an optimal trade-off in an expansive search space. In this paper, we propose to leverage recent advances in differentiable neural architecture search to discover more efficient networks. Our searched model attains 97.2% top-1 accuracy on Google Speech Command Dataset v1 with only nearly 100K parameters.

*Index Terms: Keyword spotting, neural architecture search*

(智能音频设备被一个持续运行的轻量的（以减少功耗）关键词检索程序控制。然而，设计具有高精度和低延迟的模型以实现准确和快速的响应是具有挑战性的。在开发端到端神经网络方面已经做出了许多努力，其中采用深度可分离卷积、时间卷积和lstm作为构建单元。尽管如此，这些由人类专业知识设计的网络可能无法在广阔的搜索空间中实现最佳权衡。在本文中，我们建议利用可微神经结构搜索的最新进展来发现更有效的网络。我们的搜索模型在Google Speech Command Dataset v1上获得了97.2%的top-1准确率，只有近10万个参数)

### Method
- ####  Search Space 
  我们在TC-ResNet之上设计我们的搜索空间，因为它具有出色的性能和较小的内存占用。我们还介绍了TC-ResNet块的挤压和激励(SE)模块。  
  ![](img/mk-2023-09-29-17-27-28.png)  
  具体来说，对于每个TC块，我们有{3,5,7,9}的内核大小选项，是否启用SE，以及一个额外的跳过连接。
- #### Searching Algorithm  
  放弃理解与总结，请阅读论文3.2节
  ![](img/mk-2023-09-29-17-33-14.png)  

### Experiment and Result  
#### Exprtimental Setup  
- Dataset: [google speech command datasets v1 and v2](https://arxiv.org/abs/1804.03209)  
- 其他实验设置就不记录了，复现这篇论文的可能性微乎其微。。。
#### Result
![](img/mk-2023-09-29-17-41-48.png)  
![](img/mk-2023-09-29-17-42-00.png)

### 知识补充  
#### [TC-ResNet](https://zhuanlan.zhihu.com/p/80123284)  
[Temporal Convolution for Real-time Keyword Spotting on Mobile Devices](https://arxiv.org/abs/1904.03814)  
Temporal Convolution 即在MFCC‘通道’这个维度上做一维卷积，这个一维卷积会使用到扩张卷积以扩大感受野。  
TC-ResNet：  
![](img/mk-2023-09-29-16-18-29.png)  
#### SE  
[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)   


![](img/mk-2023-09-29-16-28-58.png)
上图是我们提出的 SE 模块的示意图。给定一个输入 x，其特征通道数为 c_1，通过一系列卷积等一般变换后得到一个特征通道数为 c_2 的特征。与传统的 CNN 不一样的是，接下来我们通过三个操作来重标定前面得到的特征。

首先是 Squeeze 操作，我们顺着空间维度来进行特征压缩，将每个二维的特征通道变成一个实数，这个实数某种程度上具有全局的感受野，并且输出的维度和输入的特征通道数相匹配。它表征着在特征通道上响应的全局分布，而且使得靠近输入的层也可以获得全局的感受野，这一点在很多任务中都是非常有用的。

其次是 Excitation 操作，它是一个类似于循环神经网络中门的机制。通过参数 w 来为每个特征通道生成权重，其中参数 w 被学习用来显式地建模特征通道间的相关性。

最后是一个 Reweight 的操作，我们将 Excitation 的输出的权重看做是经过特征选择后的每个特征通道的重要性，然后通过乘法逐通道加权到先前的特征上，完成在通道维度上的对原始特征的重标定。  

SE在具体网络中的应用:  
![](img/mk-2023-09-29-16-30-17.png)

#### [NAS](https://zhuanlan.zhihu.com/p/45133026)  
(⬆这个链接指向的知乎回答写的很易懂)  
[A Comprehensive Survey of Neural Architecture Search: Challenges and Solutions](https://dl.acm.org/doi/abs/10.1145/3447582)

让程序自动的搜索出一个不错的网络架构，这一领域被称为神经架构搜索（Neural Architecture Search）。NAS的意义在于解决深度学习模型的调参问题，是结合了优化和机器学习的交叉研究。

![](img/mk-2023-09-29-16-39-19.png)
- **搜索空间（Search Space）**: 搜索空间定义了搜索的范围，其实就是在哪搜索。通过结合一些过去研究者架构设计方面的经验，可以通过减小搜索空间和简化搜索过程来提高搜索的性能。当然，这样同时也引入了人为的主观臆断，可能会妨碍寻找到超越当前人类知识的新的架构构建块（building blocks）  
  ![](img/mk-2023-09-29-16-47-38.png)![](img/mk-2023-09-29-16-49-39.png)
- **搜索策略（Search strategy）**：搜索策略定义的则怎样去搜索。一方面，我们希望能快速找到性能良好的架构，另一方面，也应避免过早收敛到次优架构（suboptimal architeture）区域。   
  到现在，已经有许多不同的搜索策略用于 NAS，主要有如下这些： 随机搜索（random search），贝叶斯优化（Bayesian optimazation），进化方法（evolutionaray methods），强化学习（Reinforcement Learning, RL），梯度方法（gradient-based methods）。 
- **性能评估策略（Performace estimation strategy）**：NAS 的目标是希望能够自动的在给定的数据集上找到一个高性能的架构。性能评估则是指评估此性能的过程：最简单的方式是按照通常的方式对一个标准架构训练和验证来获得结果，但遗憾的是这样的计算成本太高了，并且同时限制了可以搜索的网络架构的数量。因此，最近的许多研究都集中在探索新的方法来降低这些性能评估的成本。  
  ![](img/mk-2023-09-29-16-57-39.png)

#### [DARTS](https://zhuanlan.zhihu.com/p/156832334)
DARTS是第一个提出基于松弛连续化的，使用梯度下降进行搜索的神经网络架构搜索(neural architecture search， NAS)算法。

DARTS最大的贡献在于使用了Softmax对本来离散的搜索空间进行了连续化，并用类似于元学习中MAMAL的梯度近似，使得只在一个超网络上就可以完成整个模型的搜索，无需反复训练多个模型。（当然，之后基于演化算法和强化学习的NAS方法也迅速地借鉴了超网络这一特性）。  

DARTS通过以可微分的方式描述任务来解决架构搜索的可扩展性挑战。

与传统的在离散的、不可微的搜索空间上应用进化或强化学习的方法不同（这些方法需要再一堆离散的候选网络中间搜索），我们的方法基于连续松弛的结构表示，允许在验证集上使用梯度下降对结构进行高效搜索。

DARTS在大搜索空间中搜索有复杂拓扑结构的高效架构cell，而过去使用梯度下降更多是找滤波器形状，分支方式之类的低维超参数。
![](img/mk-2023-09-29-17-06-15.png)  
> 在CNN的DARTS中，可选的操作集合为：3×3深度可分离卷积，5×5深度可分离卷积，3×3空洞深度可分离卷积，5×5空洞深度可分离卷积，3×3极大值池化，3×3均值池化，恒等，0操作（两个节点直接无连接）

![](img/mk-2023-09-29-17-09-30.png)  
![](img/mk-2023-09-29-17-10-09.png)  
![](img/mk-2023-09-29-17-11-01.png)  
![](img/mk-2023-09-29-17-11-14.png)


***
## 4. Convmixer: Feature interactive convolution with curriculum learning for small footprint and noisy far-field keyword spotting  
***留意此文***
> Ng D, Chen Y, Tian B, et al. Convmixer: Feature interactive convolution with curriculum learning for small footprint and noisy far-field keyword spotting[C]//ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022: 3603-3607.(阿里巴巴，新加坡南洋理工)  

### Abstract  
Building efficient architecture in neural speech processing is paramount to success in keyword spotting deployment. However, it is very challenging for lightweight models to achieve noise robustness with concise neural operations. In a realworld application, the user environment is typically noisy and may contain reverberations. We proposed a novel feature interactive convolutional model with merely 100K parameters to tackle this under the noisy far-field condition. The interactive unit is proposed in place of the attention module that promotes the flow of information with more efficient computations. Moreover, curriculum-based multi-condition training is adopted to attain better noise robustness. Our model achieves 98.2% top-1 accuracy on Google Speech Command V2-12 and is competitive against large transformer models under the designed noise condition.  

*Index Terms: keyword spotting, small footprint, noisy far-field*  

(构建高效的神经语音处理体系结构是关键字识别部署成功的关键。然而，对于轻量级模型来说，用简洁的神经运算实现噪声鲁棒性是非常具有挑战性的。在实际应用程序中，用户环境通常是嘈杂的，并且可能包含混响。为了解决这一问题，我们提出了一种仅包含100K个参数的特征交互卷积模型。提出以交互单元代替注意力模块，以更高效的计算促进信息的流动。此外，采用基于课程的多条件训练，获得更好的噪声鲁棒性。我们的模型在Google Speech Command V2-12上达到了98.2%的top-1准确率，并且在设计噪声条件下与大型变压器模型具有竞争力。)  

### Method  
#### Model
![](img/mk-2023-09-30-10-08-57.png)  

ConvMixer网络由三个主要部分组成: 
- **pre-convolutional block**:  
  1D DS Convolution + BN + Swish
- **convolution-mixer block**:  
  ConvMixer块将前一个信道x时间特征传递给二维卷积子块进行频域提取。这创建了第三个维度，表达了来自频域的丰富信息。为了保持之前输入的形状，我们使用了一个逐点卷积，将其压缩回适合形状。然后，利用一维DWS块实现时域特征提取。这两种操作的乘积将产生频率和时间丰富的嵌入。接下来，我们构建了一个混频器层，以允许信息在全局特征通道上流动。最后，我们添加了先前输出的跳过连接和连接到块输出的2D特征。  
  ![](img/mk-2023-09-30-10-29-43.png)  
  ![](img/mk-2023-09-30-10-30-06.png)  
  利用两种类型的多层感知器(MLP)，即时间信道混合和频率信道混合，来诱导特征空间之间的相互作用。每个MLP混合涉及两个线性层和一个独立于每个时间和频率通道的GELU激活单元。  
  ![](img/mk-2023-09-30-10-32-26.png)  
  ![](img/mk-2023-09-30-10-32-44.png)
- **postconvolutional block**:  
  1D DS Convolution + BN + Swish

####  Curriculum Based Multi-condition Training  
![](img/mk-2023-09-30-13-49-38.png)  

我们将训练过程分成五个难度逐渐加大的步骤。一开始，我们在没有噪声的干净样本上调节模型。在接下来的三个步骤中，将以-5dB的增量向固定的N个样本中引入噪声，并且N个样本中的所有条件均匀分布，即[clean, 0]， [clean, 0， -5]， [clean, 0， -5， -10]。最后，我们通过用房间脉冲响应(RIR)数据增加一半的数据集来包括远场音频。  

在每个阶段的每个时代，我们用验证精度和损失记录学习进度。接下来，将进度步距准则c定义为归一化验证精度与损失之间的差。归一化是基于之前时代的准确性和损失。下方公式描述了计算归一化精度和损失的第m个历元值的一般算法。注意，如果m等于零，则归一化结果为零。随后，如果c连续10个epoch不高于当前最佳准则，则加载最新最佳准则的模型，进入下一难度阶段进行训练。  
![](img/mk-2023-09-30-13-51-21.png)  

### Experiment and Result  
#### Exprtimental Setup  
- Dataset: [google speech command datasets v2](https://arxiv.org/abs/1804.03209)  
- Noise Dataset: [MUSAN](https://arxiv.org/abs/1510.08484)
- Far-field Dataset: [ BUT Speech@FIT Reverberation Database](https://ieeexplore.ieee.org/abstract/document/8717722)
- Features: FBank, 25ms of window, 10ms of overlop。我们将FBank的分辨率固定为98 × 64，相当于1s的语音。短于1s的命令将在右侧补零。在训练期间，使用-100到100ms范围内的时移来执行数据增强。采用最大长度为25的时间和频率掩蔽参数对频谱图进行掩蔽。我们使用从[0，-5，-10]dB列表中选择的信噪比生成噪声数据。然后，为了获得更强的学习正则化，对训练样本执行混合输入，混合比为0.5。
- batch size of 128,  initial learning rate of 6e-3 factored by 0.85 on every
four epoch intervals after the fifth epoch, Adam optimizer, binary cross-entropy loss, 200 epochs  
#### Result
![](img/mk-2023-09-30-14-24-11.png)

### 知识补充  
#### Swish  
公式：*Swish(x)=x∗Sigmoid(x)*    

![](img/mk-2023-09-30-13-00-45.png)
![](img/mk-2023-09-30-13-01-08.png)

优点：  
- 有助于防止慢速训练期间，梯度逐渐接近0并导致饱和
- 导数恒大于0。
- 平滑度在优化和泛化中起了重要作用。

#### MLP Mixer  
[MLP-Mixer: An all-MLP Architecture for Vision](https://proceedings.neurips.cc/paper/2021/hash/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Abstract.html)  
MLP-Mixer将CNN这两个任务切割开来，用两个MLP网络来处理  

model overview:  

![](img/mk-2023-09-30-13-16-06.png)  

先将输入图片拆分成patches，然后通过Per-patch Fully-connected将每个patch转换成feature embedding，然后送入N个Mixer Layer，最后通过Fully-connected进行分类。

Mixer分为channel-mixing MLP和token-mixing MLP两类。channel-mixing MLP允许不同通道之间进行交流；token-mixing MLP允许不同空间位置(tokens)进行交流。这两种类型的layer是交替堆叠的，方便支持两个输入维度的交流。每个MLP由两层fully-connected和一个GELU构成。

mixer architecture:

![](img/mk-2023-09-30-13-17-51.png)

Mixer结构如上图所示。每个Mixer结构由两个MLP blocks构成，其中红色框部分是token-mixing MLP，绿色框部分是channel-mixing MLP。

token-mixing MLP block作用在X的列上(即先对X进行转置)，并且所有列参数共享MLP1，得到的输出重新转置一下。

channel-mixing MLP block作用在行上，所有行参数共享MLP2。

#### [Curriculum Learning](https://zhuanlan.zhihu.com/p/362351969)  
[A Survey on Curriculum Learning](https://ieeexplore.ieee.org/abstract/document/9392296)  

课程学习 (Curriculum learning, CL) 是近几年逐渐热门的一个前沿方向。它是一种训练策略，模仿人类的学习过程，主张让模型先从容易的样本开始学习，并逐渐进阶到复杂的样本和知识。CL策略在计算机视觉和自然语言处理等多种场景下，在提高各种模型的泛化能力和收敛率方面表现出了强大的能力。

课程学习的核心问题是得到一个ranking function，该函数能够对每条数据/每个任务给出其learning priority (学习优先程度)。这个则由**难度测量器**（Difficulty Measurer）实现。另外，我们什么时候把 Hard data 输入训练 以及 每次放多少呢？ 这个则由**训练调度器** （Training Scheduler）决定。因此，目前大多数CL都是基于"难度测量器+训练调度器 "的框架设计。根据这两个**是否自动设计**可以将CL分成两个大类即 **Predefined CL** 和 **Automatic CL**。

Predifined CL 的难度测量器和训练调度器都是利用人类先验先验知识由人类专家去设计；  
Automatic CL 的至少其中一个是以数据驱动的方式自动设计。

![](img/mk-2023-09-30-13-44-22.png)  
![](img/mk-2023-09-30-13-44-38.png)  
![](img/mk-2023-09-30-13-45-19.png)  
![](img/mk-2023-09-30-13-45-38.png)  
#### [深度学习模型计算量评价指标FLOPs, MACs, MAdds关系](http://t.csdnimg.cn/0HPpL)

***
## 5. Dynamic curriculum learning via data parameters for noise robust keyword spotting
> Higuchi T, Saxena S, Souden M, et al. Dynamic curriculum learning via data parameters for noise robust keyword spotting[C]//ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021: 6848-6852.(苹果)  

### Abstract  
We propose dynamic curriculum learning via data parameters for noise robust keyword spotting. Data parameter learning has recently been introduced for image processing,
where weight parameters, so-called data parameters, for target classes and instances are introduced and optimized along with model parameters. The data parameters scale logits and control importance over classes and instances during training, which enables automatic curriculum learning without additional annotations for training data. Similarly, in this paper, we propose using this curriculum learning approach
for acoustic modeling, and train an acoustic model on clean and noisy utterances with the data parameters. The proposed approach automatically learns the difficulty of the classes and instances, e.g. due to low speech to noise ratio (SNR), in the gradient descent optimization and performs curriculum learning. This curriculum learning leads to overall improvement of the accuracy of the acoustic model. We evaluate the effectiveness of the proposed approach on a keyword spotting task. Experimental results show 7.7% relative reduction in false reject ratio with the data parameters compared to a baseline model which is simply trained on the multiconditioned dataset.  

Index Terms: Noise robustness, acoustic modeling, keyword spotting, curriculum learning

（我们提出动态课程学习通过数据参数噪声鲁棒关键字发现。数据参数学习最近被引入到图像处理中，其中目标类和实例的权重参数，即所谓的数据参数，与模型参数一起被引入和优化。数据参数在训练过程中缩放逻辑并控制类和实例的重要性，从而实现自动课程学习，而无需对训练数据进行额外的注释。同样，在本文中，我们建议使用这种课程学习方法进行声学建模，并使用数据参数训练干净和有噪声的话语声学模型。该方法在梯度下降优化中自动学习类和实例的难度，例如由于语音噪声比(SNR)低，并进行课程学习。本课程的学习使声学模型的准确性得到全面提高。我们评估了所提出的方法在关键字发现任务上的有效性。实验结果表明，与在多条件数据集上简单训练的基线模型相比，该模型的误拒率相对降低了7.7%。）

这篇真的有点没太看懂，相关资料也少，今天也不是特别专注，啃不下去。



***
## 6. End-to-end keyword spotting using neural architecture search and quantization  
> Peter D, Roth W, Pernkopf F. End-to-end keyword spotting using neural architecture search and quantization[C]//ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022: 3423-3427.(奥地利格拉茨理工大学)

### Abstract
This paper introduces neural architecture search (NAS) for the automatic discovery of end-to-end keyword spotting (KWS) models for limited resource environments. We employ a differentiable NAS approach to optimize the structure of convolutional neural networks
(CNNs) operating on raw audio waveforms. After a suitable KWS model is found with NAS, we conduct quantization of weights and activations to reduce the memory footprint. We conduct extensive experiments on the Google speech commands dataset. In particular, we compare our end-to-end models to mel-frequency cepstral coefficient (MFCC) based CNNs. For quantization, we compare fixed bitwidth quantization and trained bit-width quantization. Using NAS only, we were able to obtain a highly efficient model with an accuracy of 95.55% using 75.7k parameters and 13.6M operations. Using trained bit-width quantization, the same model achieves a test accuracy of 93.76% while using on average only 2.91 bits per activation and 2.51 bits per weight.

Index Terms: keyword spotting, neural architecture search, quantization

（本文引入神经结构搜索(NAS)，用于有限资源环境下的端到端关键字识别(KWS)模型的自动发现。我们采用一种可微的NAS方法来优化卷积神经网络(cnn)在原始音频波形上的结构。在使用NAS找到合适的KWS模型后，我们对权重和激活进行量化以减少内存占用。我们在谷歌语音命令数据集上进行了大量的实验。特别地，我们将我们的端到端模型与基于mel频率倒谱系数(MFCC)的cnn进行了比较。对于量化，我们比较了固定位宽量化和训练位宽量化。仅使用NAS，我们就能够使用75.7k个参数和136m个操作获得准确率为95.55%的高效模型。使用经过训练的位宽量化，相同的模型在每次激活平均仅使用2.91比特和每个权重平均仅使用2.51比特的情况下，达到了93.76%的测试精度。）

### Method  
#### Neural Architecture Search

![](img/mk-2023-09-30-16-48-37.png)  
使用ProxylessNAS搜索最优模型。

#### Feature Extraction using SincConvs
使用SincNet对**原始音频**进行特征提取

#### Weight and Activation Quantization
使用固定位宽量化或训练位宽量化对模型进行压缩。

### Experiment and Result  
#### Exprtimental Setup  
- Dataset: [google speech command datasets v1](https://arxiv.org/abs/1804.03209)
#### Result
![](img/mk-2023-09-30-17-06-53.png)

### 知识补充
#### [ProxyLessNAS](https://zhuanlan.zhihu.com/p/144318917)
[ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://openreview.net/forum?id=HylVB3AqYm)  
[ProxyLessNAS Code](https://github.com/MIT-HAN-LAB/ProxylessNAS)  

![](img/mk-2023-09-30-17-13-47.png)
![](img/mk-2023-09-30-17-19-53.png)
暂时不深入细节了。。。。。

#### SincNet  
[Interpretable Convolutional Filters with SincNet](https://arxiv.org/abs/1811.09725)  
[Pytorch-SincNet Code](https://github.com/mravanelli/SincNet/)
![](img/mk-2023-09-30-17-23-49.png)  
![](img/mk-2023-09-30-17-24-12.png)  
![](img/mk-2023-09-30-17-25-16.png)

#### Quantization  
模型量化是一种模型压缩方式
![](img/mk-2023-09-30-17-46-14.png)  

正如其它模型压缩方法一样，对模型的量化基于一个共识。那就是复杂的、高精度表示的模型在训练时是必要的，因为我们需要在优化时捕捉微小的梯度变化，然而在推理时并没有必要。也就是说，网络中存在很多不重要的参数，或者并不需要太细的精度来表示它们。另外，实验证明神经网络对噪声鲁棒，而量化其实也可看作是噪声。这就意味着我们在部署模型前可以将之化简，而表示精度降低就是化简的重要手段之一。我们知道，大多深度学习训练框架默认下模型的参数是32位浮点的，计算也是32位浮点的。模型量化的基本思想就是用更低精度（如8位整型）来代替原浮点精度。听起来似乎非常的简单，但是细看之下会发现这个坑比想象中大得多。从相关的文献可以看到各大巨头公司都或多或少地参于其中，似乎成为了兵家必争之地。量化最核心的挑战是如何在减少表示精度的同时不让模型的准确度掉下来，即在压缩率与准确率损失间作trade-off。这就衍生出很多有趣的子问题，比如量化对象是什么（weight，activation，gradient），量化到几位（8位，4位，2位，1位），量化参数（如step size，clipping value）如何选择，量化参数是否可以自动优化，不同层是否需要不同的量化参数，如何在量化后恢复准确率或者在训练时考虑量化，等等。。。  

![](img/mk-2023-09-30-17-49-50.png)  
![](img/mk-2023-09-30-17-51-16.png)
![](img/mk-2023-09-30-17-51-57.png)
![](img/mk-2023-09-30-17-52-51.png)
pytorch对量化的支持：
![](img/mk-2023-09-30-17-53-57.png)

#### Brevitas  
Brevitas是一个用于量化感知训练（QAT）的Pytorch库。  
[Brevitas Code](https://github.com/Xilinx/brevitas)  
布雷维塔斯目前正在积极开发中。文档、测试、示例和预训练模型将逐步发布。

请注意，Brevitas是一个研究项目，而不是Xilinx的官方产品。


***
