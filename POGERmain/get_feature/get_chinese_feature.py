import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# 加载停用词列表
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f.readlines()])


# 文本预处理函数
def preprocess_text(text):
    # 去除标点符号和特殊字符
    text = re.sub(r'[^\u4e00-\u9fa5\w\s]', '', text)
    # 分词（注意：这里实际上TF-IDF向量化器会进行自己的分词，但我们可以先做一些基本的处理）
    tokens = text.split()
    # 去除停用词
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return ' '.join(filtered_tokens)


# 加载数据
texts = []
with open('test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for text in data['text']:  # 假设'text'是一个列表，包含所有行的文本
        texts.append(text)

    # 将所有文本合并为一个长字符串
merged_text = ' '.join(texts)

# 初始化TF-IDF向量化器
# 注意：这里我们直接在向量化器中处理分词和停用词，而不是在预处理函数中
# 但为了演示，我们还是先通过预处理函数处理文本，然后传递给向量化器（尽管这样做可能不是最高效的）
# 如果你想要让TF-IDF向量化器自己处理分词，可以省略preprocess_text步骤，并直接在向量化器中设置tokenizer参数
# 但对于中文，sklearn的TF-IDF向量化器默认不支持分词，你可能需要使用jieba等分词库与CountVectorizer结合
vectorizer = TfidfVectorizer(tokenizer=lambda doc: preprocess_text(doc).split(), lowercase=False)

# 计算TF-IDF
tfidf_matrix = vectorizer.fit_transform([merged_text])

# 初始化一个字典来存储每个词的TF-IDF值
word_tfidf = defaultdict(float)
feature_names = vectorizer.get_feature_names_out()
for word_idx, word in enumerate(feature_names):
    tfidf_score = tfidf_matrix[0, word_idx].toarray().flatten()[0]
    word_tfidf[word] = tfidf_score