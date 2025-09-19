#!/usr/bin/env python
import nltk
from nltk.corpus import brown
from collections import Counter

# ====================== 1. 加载语料库 ======================
# 确保已下载 Brown 语料库：nltk.download('brown')
sentences = brown.sents()  # 获取所有已分词的句子，格式：[['The', 'Fulton', ...], ...]


# ====================== 2. 构建 n-gram 统计 ======================
def build_ngrams(sentences, n):
    """
    构建 n-gram 统计：
    - n=1: unigram（单个词频率）
    - n=2: bigram（连续两个词的频率）
    - n=3: trigram（连续三个词的频率）
    """
    ngrams = []
    for sent in sentences:
        # 对每个句子，填充起始符 <s>，保证 n-gram 能覆盖句子开头
        # 比如 trigram 需要在句首补 2 个 <s>，变成 [<s>, <s>, 词1, 词2, ...]
        padded_sent = ["<s>"] * (n - 1) + list(sent)
        # 生成 n-gram，如 trigram 会生成 (词1, 词2, 词3), (词2, 词3, 词4) 等
        ngrams.extend(list(nltk.ngrams(padded_sent, n)))
    # 用 Counter 统计每个 n-gram 的出现次数
    return Counter(ngrams)

# 构建 unigram、bigram、trigram
unigrams = build_ngrams(sentences, 1)
bigrams = build_ngrams(sentences, 2)
trigrams = build_ngrams(sentences, 3)


# ====================== 3. 实现预测逻辑 ======================
def predict_next_word(context, ngram_counter, n, top_k=5):
    """
    根据 context 预测下一个词，逻辑：
    1. 处理 context，补全起始符 <s>（适配 n-gram 的长度）
    2. 提取 n-gram 的“前缀”（比如 trigram 提取最后 2 个词 + <s>）
    3. 筛选所有以该前缀开头的 n-gram，取最后一个词作为候选
    4. 按 n-gram 出现次数排序，返回 top-k 候选词
    """
    # 拆分 context 为列表，如 "I want to eat" → ["I", "want", "to", "eat"]
    context_words = context.split()
    # 补全起始符，让前缀长度 = n-1（比如 trigram 需补 2 个 <s>，但如果 context 较长，可能不需要补）
    needed_pads = max(0, (n - 1) - len(context_words))
    padded_context = ["<s>"] * needed_pads + context_words
    # 提取前缀（n-gram 的前 n-1 个词），比如 trigram 取最后 2 个词 + <s>
    prefix = tuple(padded_context[-(n - 1):])

    # 筛选所有以 prefix 为前缀的 n-gram，取最后一个词作为候选
    candidates = []
    for ngram, count in ngram_counter.items():
        if ngram[:-1] == prefix:  # ngram[:-1] 是前缀，ngram[-1] 是下一个词
            candidates.append((ngram[-1], count))  # (候选词, 出现次数)

    # 按出现次数降序排序，取 top-k
    candidates.sort(key=lambda x: -x[1])
    return [word for word, _ in candidates[:top_k]]


# ====================== 4. 测试预测功能 ======================
if __name__ == "__main__":
    # 测试输入（已分词，空格分隔）
    context = "I want to eat"

    # 分别用 unigram、bigram、trigram 预测
    # 注意：unigram 不依赖上下文，所以预测结果是语料库中最常见的词
    pred_unigram = predict_next_word(context, unigrams, 1, top_k=3)
    pred_bigram = predict_next_word(context, bigrams, 2, top_k=3)
    pred_trigram = predict_next_word(context, trigrams, 3, top_k=3)

    # 输出结果
    print(f"Unigram 预测结果（最常见词）：{pred_unigram}")
    print(f"Bigram 预测结果（基于最后1个词 'eat' 前的 'to' ）：{pred_bigram}")
    print(f"Trigram 预测结果（基于最后2个词 'to eat' 前的 'want to' ）：{pred_trigram}")
