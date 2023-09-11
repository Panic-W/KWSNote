# The NPU System for the 2020 Personalized Voice Trigger Challenge

## 摘要



## 方法



## 实验设置：

### 标签怎么打？

For a positive training utterance, we select up-to **40 frames** around middle frame of the WuW region as positive training samples and assign 1 to them. Other frames in the positive training utterance are discarded as ambiguous and are not used in training. For negative training utterances, all frames are regarded as negative training samples and assigned to 0. Our KWS system is thus modeled as a sequence binary classification problem. To train the model, binary cross entropy (BCE) loss is used

### 数据增强怎么增强？

#### 1、调整关键词出现在语音中位置

The positive keyword training set is composed as the following parts. 

- 1) The keyword segments in the training positive utterances; 
- 2) Randomly select non-keyword speech segments and pad them before the above keyword segments in 1); 
- 3) Pad non-keyword speech segments both before and after the above keyword segments in 1);

 In addition, we also create more negative training utterances. The specific strategy is to cut the positive utterance in 3) at the middle frame of the keyword into two segments which are subsequently used as negative training examples. This kind of negative examples can improve the generalization ability of the model too. 

#### 2、对声谱图进行增强（在时域和频域加入掩码）

SpecAugment [2] is also applied during training, which is first proposed for end-to-end (E2E) ASR to alleviate over-fitting and has recently proven to be effective in training E2E KWS system as well [3]. Specifically, we **apply time as well as freqency masking** during training. We randomly select 0 − 20 consecutive frames and set all of their Mel-filter banks to zero, for time masking. For frequency masking, we randomly select 0 − 30 consecutive dimensions of the 80 Mel-filter banks and set their values to zero for all frames of the utterance. 

### 估计关键词出现的位置

As mentioned before, the keywords always appear at the end of positive utterances. Based on this, we do not explicitly predict the starting and ending frames of the keywords. Instead, in an utterance, we take the frame with the **largest keyword posterior as the middle position of the keyword**. We use the estimated middle position and the end frame of the keyword to estimate the starting frame of the keyword. Although this trick can not be applied to real applications, it is effective for the specific condition of this challenge. Note that there are several previous studies that explicitly model the location of keyword in a positive utterance [4, 5, 6, 7].

