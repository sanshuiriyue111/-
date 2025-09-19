#!/usr/bin/env python
import nltk
from nltk.corpus import brown
from gensim.models import Word2Vec
import numpy as np

sentences = brown.sents()
#加载语料库后，进行简单的预处理（分词）

#接下来训练Word2Vec 模型
model = Word2Vec(sentences,sg=0,vector_size=100,window=5,min_count=5);
#sg=0表示使用CBOW架构，上下文窗口大小为5，min_count是忽略出现次数小于该值的单词；

#查询与“computer”相似的词：
if "computer" in model.wv:
  similar_words = model.wv.most_similar("computer",topn=5)
  print("与‘computer’最相似的5个词")
  for word,similarity in similar_words:
    print(f"{word}: {similarity:.4f}")
else:
  print("语料库中没有‘computer’这个词")

#计算两个词的相似度：
word1 = "apple"
word2 = "fruit"
if word1 in model.wv and word2 in model.wv:
  similarity = model.wv.similarity(word1,word2)
  print(f"'{word1}' 和 '{word2}' 的相似度：{similarity:.4f}")
else:
  print(f"语料库中缺少 '{word1}' 或 '{word2}'")
  
#词向量的算术运算，以某经典例子验证：
words_to_check = ["king","man","woman"]
if all(word in model.wv for word in words_to_check):
  result = model.wv.most_similar(positive=["king","woman"],negative=["man"],topn=1)
  print(f"'king' - 'man' + 'woman' 最接近的词：{result[0][0]}（相似度：{result[0][1]:.4f}）")
else:
  print("语料库中缺少进行该运算所需的词汇")
  

