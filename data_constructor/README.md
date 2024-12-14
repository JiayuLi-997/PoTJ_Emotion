# Data constructor

本文件夹下为生成时间相关微博，并进行情绪预测的相关代码。

./corpus_constructor中，包含词典匹配提取（Words_match.py）和句典匹配提取（Sentence_match.py），及最终merge为统一corpus（MergeAll.ipynb）

./sentiment_analysis中，包含基于pretrained BERT的valence prediction（bert_valence_predict.py）和基于word embedding的离散情绪prediction(discrete_predict.ipynb)，及整理为最终文件的get_final_data.ipynb