## Mamba: Linear-Time Sequence Modeling with Selective State Spaces  
>Gu A, Dao T. Mamba: Linear-time sequence modeling with selective state spaces[J]. arXiv preprint arXiv:2312.00752, 2023.  

### Abstract  
基础模型现在为深度学习中大多数令人兴奋的应用程序提供动力，几乎普遍基于Transformer架构及其核心注意力模块。许多亚二次时间架构（如线性注意力、门控卷积和递归模型以及结构化状态空间模型（SSMs））已被开发用于解决变压器在长序列上的计算效率低下问题，但它们在重要模态（如语言）上的表现不如注意力。我们发现这种模型的一个主要缺点是无法执行基于内容的推理，并做出了一些改进。首先，简单地让SSM参数是输入的函数解决了它们在离散模态中的弱点，允许模型根据当前令牌沿着序列长度维度选择性地传播或遗忘信息。第二，尽管这种变化阻止了有效卷积的使用，但我们设计了递归模式下的硬件感知并行算法。我们将这些选择性SSM集成到一个简化的端到端神经网络架构中，无需关注甚至无需MLP块（曼巴）。Mamba具有快速推理能力（吞吐量比Transformers高5倍）和序列长度的线性伸缩性，其性能在长达百万长度序列的真实数据上有所提高。作为通用序列模型主干，Mamba在语言、音频和基因组学等多种模态上实现了一流的性能。在语言建模方面，我们的曼巴-3B模型在预训练和下游评估方面都优于相同大小的变压器，并且匹配两倍大小的变压器。  

### Introduction  























