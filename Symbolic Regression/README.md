# 符号回归方法集

此存储库包含精选的符号回归方法列表，包括其实现、相关研究论文以及每种方法的其他资源。

## 概述

符号回归是一种回归分析，用于寻找最适合给定数据集的数学表达式。与将数据拟合到预定义方程的传统回归模型不同，符号回归会找到方程本身的函数形式。

## 用法

要使用特定的符号回归方法，请导航到该方法的目录并按照该方法子文件夹中提供的具体说明进行操作。

## 方法

|   Method   |                            Title                             |                           Authors                            | Year |                             Code                             |                Publications                |
| :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--: | :----------------------------------------------------------: | :----------------------------------------: |
|    TPSR    | [Transformer-based Planning for Symbolic Regression](https://arxiv.org/abs/2303.06833) | [Parshin Shojaee](https://arxiv.org/search/cs?searchtype=author&query=Shojaee,+P), [Kazem Meidani](https://arxiv.org/search/cs?searchtype=author&query=Meidani,+K), et al. | 2023 | [code](https://github.com/deep-symbolic-mathematics/TPSR?tab=readme-ov-file) |                  NeurIPS                   |
|    E2E     | [End-to-end symbolic regression with transformers](https://arxiv.org/abs/2204.10532) | [Pierre-Alexandre Kamienny](https://arxiv.org/search/cs?searchtype=author&query=Kamienny,+P), [Stéphane d'Ascoli](https://arxiv.org/search/cs?searchtype=author&query=d'Ascoli,+S), et al. | 2022 | [code](https://github.com/facebookresearch/symbolicregression) |                  NeurIPS                   |
|   NSRTS    | [Neural Symbolic Regression that Scales](https://arxiv.org/abs/2106.06427) | [Luca Biggio](https://arxiv.org/search/cs?searchtype=author&query=Biggio,+L), [Tommaso Bendinelli](https://arxiv.org/search/cs?searchtype=author&query=Bendinelli,+T), et al. | 2021 | [code](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales) |                    ICML                    |
|    DSR     | [Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients](https://openreview.net/forum?id=m5Qsh0kBQG) | [Brenden K Petersen](https://openreview.net/profile?id=~Brenden_K_Petersen1)*,* [Mikel Landajuela Larma](https://openreview.net/profile?email=landajuelala1@llnl.gov), et al. | 2021 | [code](https://github.com/dso-org/deep-symbolic-optimization?tab=readme-ov-file) |                    ICLR                    |
| AI-Feynman | [AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity](https://arxiv.org/abs/2006.10782) | [Silviu-Marian Udrescu](https://arxiv.org/search/cs?searchtype=author&query=Udrescu,+S), [Andrew Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan,+A), et al. | 2020 |         [code](https://github.com/lacava/AI-Feynman)         |                  NeurIPS                   |
|    DGSR    | [Deep Generative Symbolic Regression](https://openreview.net/pdf?id=o7koEEMA1bR) | [Samuel Holt](https://openreview.net/profile?id=~Samuel_Holt1)*,* [Zhaozhi Qian](https://openreview.net/profile?id=~Zhaozhi_Qian1), et al. | 2023 |                           [code]()                           |                    ICLR                    |
|    BSR     | [Bayesian Symbolic Regression](https://arxiv.org/abs/1910.08892) | [Ying Jin](https://arxiv.org/search/stat?searchtype=author&query=Jin,+Y), [Weilin Fu](https://arxiv.org/search/stat?searchtype=author&query=Fu,+W), et al. | 2020 |        [code](https://github.com/ying531/MCMC-SymReg)        |                     —                      |
|    ITEA    | [Interaction–Transformation Evolutionary Algorithm for Symbolic Regression](https://arxiv.org/abs/1902.03983) | [Fabricio Olivetti de Franca](https://arxiv.org/search/cs?searchtype=author&query=de+Franca,+F+O), [Guilherme Seidyo Imai Aldeia](https://arxiv.org/search/cs?searchtype=author&query=Aldeia,+G+S+I) | 2020 |          [code](https://github.com/folivetti/ITEA/)          |          Evolutionary Computation          |
|   Operon   | [Operon C++: an efficient genetic programming framework for symbolic regression](https://dl.acm.org/doi/10.1145/3377929.3398099) | [Bogdan Burlacu](https://pure.fh-ooe.at/en/persons/bogdan-burlacu), [Gabriel Kronberger](https://pure.fh-ooe.at/en/persons/gabriel-kronberger), et al. | 2020 |       [code](https://github.com/heal-research/operon)        |                   GECCO                    |
|  GPGOMEA   | [Scalable genetic programming by gene-pool optimal mixing and input-space entropy-based building-block learning](https://dl.acm.org/doi/10.1145/3071178.3071287) |          Marco Virgolin, Tanja Alderliesten, et al.          | 2017 |      [code](https://github.com/marcovirgolin/GP-GOMEA/)      |                   GECCO                    |
|   SBPGP    | [Linear scaling with and within semantic backpropagation-based genetic programming for symbolic regression](https://dl.acm.org/doi/10.1145/3321707.3321758) |          Marco Virgolin, Tanja Alderliesten, et al.          | 2019 |      [code](https://github.com/marcovirgolin/GP-GOMEA)       |                   GECCO                    |
|  gplearn   |                              —                               |                              —                               |  —   |      [code](https://github.com/trevorstephens/gplearn)       |                     —                      |
|    FEAT    | [Learning concise representations for regression by evolving networks of trees](https://openreview.net/pdf?id=Hke-JhA9Y7) | [William La Cava](https://arxiv.org/search/cs?searchtype=author&query=La+Cava,+W), [Tilak Raj Singh](https://arxiv.org/search/cs?searchtype=author&query=Singh,+T+R), et al. | 2019 |            [code](https://github.com/lacava/feat)            |                    ICLR                    |
|    FFX     | [FFX: Fast, Scalable, Deterministic Symbolic Regression Technology](https://link.springer.com/chapter/10.1007/978-1-4614-1770-5_13) | [Trent McConaghy](https://link.springer.com/chapter/10.1007/978-1-4614-1770-5_13#auth-Trent-McConaghy) | 2011 |           [code](https://github.com/natekupp/ffx)            | Genetic Programming Theory and Practice IX |
| AFP/FPAFP  | [Age-fitness pareto optimization](https://doi.org/10.1145/1830483.1830584) |         Age-fitness pareto optimization, Hod Lipson          | 2009 |           [code](https://github.com/cavalab/ellyn)           |                   GECCO                    |



## 贡献

欢迎向此存储库做出贡献。请参阅 CONTRIBUTING.md 文件以了解有关如何做出贡献的更多详细信息。

## 执照

该项目采用 MIT 许可证 - 有关详细信息，请参阅 LICENSE 文件。

## 致谢

我们感谢所有研究人员和开发人员，是他们的贡献使得这一合集成为可能。