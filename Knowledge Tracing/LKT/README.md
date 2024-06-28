LKT
=====
LBKT+FKT集成模型，将时间增强和多种学习行为因子（响应时间、提示次数、尝试次数）融合起来。
核心架构为循环神经网络+transformer


Dependencies
----
- numpy==1.22.4
- scikit_learn==1.3.2
- torch==1.13.0+cu117
- pandas==2.0.3
- tqdm==4.65.0

Usage
----
Run LKT.py
```
python LKT.py
```

```
# 模型参数
# ===========================================================================
# Layer (type:depth-idx)                             Param #
# ===========================================================================
# Model_exp                                          --
# ├─EncoderEmbedding: 1-1                            --
# │    └─Embedding: 2-1                              9,088,512
# │    └─Embedding: 2-2                              5,120
# │    └─Embedding: 2-3                              51,200
# │    └─Embedding: 2-4                              1,024
# │    └─Embedding: 2-5                              20,480
# ├─MyTransformerEncoder: 1-2                        --
# │    └─ModuleList: 2-6                             --
# │    │    └─MyTransformerEncoderLayer: 3-1         1,576,452
# │    │    └─MyTransformerEncoderLayer: 3-2         1,576,452
# │    │    └─MyTransformerEncoderLayer: 3-3         1,576,452
# │    │    └─MyTransformerEncoderLayer: 3-4         1,576,452
# │    │    └─MyTransformerEncoderLayer: 3-5         1,576,452
# │    │    └─MyTransformerEncoderLayer: 3-6         1,576,452
# │    └─LayerNorm: 2-7                              1,024
# ├─DecoderEmbedding: 1-3                            --
# │    └─Embedding: 2-8                              9,088,512
# │    └─Embedding: 2-9                              88,064
# │    └─Embedding: 2-10                             51,200
# │    └─Embedding: 2-11                             1,024
# │    └─Embedding: 2-12                             20,480
# ├─MyTransformerDecoder: 1-4                        --
# │    └─ModuleList: 2-13                            --
# │    │    └─MyTransformerDecoderLayer: 3-7         2,626,581
# │    │    └─MyTransformerDecoderLayer: 3-8         2,626,581
# │    │    └─MyTransformerDecoderLayer: 3-9         2,626,581
# │    │    └─MyTransformerDecoderLayer: 3-10        2,626,581
# │    │    └─MyTransformerDecoderLayer: 3-11        2,626,581
# │    │    └─MyTransformerDecoderLayer: 3-12        2,626,581
# │    └─LayerNorm: 2-14                             1,024
# ├─LBKTcell: 1-5                                    198,276
# │    └─Layer1: 2-15                                32,896
# │    └─Layer1: 2-16                                32,896
# │    └─Layer1: 2-17                                32,896
# │    └─Linear: 2-18                                52,096
# │    └─Dropout: 2-19                               --
# │    └─Linear: 2-20                                82,048
# │    └─Sigmoid: 2-21                               --
# ├─Linear: 1-6                                      9,106,263
# ├─Sequential: 1-7                                  --
# │    └─Linear: 2-22                                262,400
# │    └─ReLU: 2-23                                  --
# │    └─Linear: 2-24                                10,280
# ├─Sequential: 1-8                                  --
# │    └─Linear: 2-25                                41
# │    └─Sigmoid: 2-26                               --
# ├─Sequential: 1-9                                  --
# │    └─Linear: 2-27                                65,664
# │    └─ReLU: 2-28                                  --
# │    └─Dropout: 2-29                               --
# │    └─Linear: 2-30                                1,290
# ├─Linear: 1-10                                     5,632
# ├─LayerNorm: 1-11                                  1,024
# ├─Sigmoid: 1-12                                    --
# ===========================================================================
# Total params: 53,519,564
# Trainable params: 53,519,564
# Non-trainable params: 0
# ===========================================================================
```




