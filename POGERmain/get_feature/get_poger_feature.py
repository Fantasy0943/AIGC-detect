import os
import sys
import json
import math
import nltk
import torch
import argparse
import tiktoken
import numpy as np
import concurrent.futures

from tqdm import tqdm
from openai import OpenAI
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer


def get_root_word(word):
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer("english")
    return stemmer.stem(word)


class GPT2Estimator:

    def __init__(self, model_name_or_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, pad_token_id=self.tokenizer.eos_token_id).to(device)

    def estimate(self, context, n, temp, max_token):
        input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=max_token, do_sample=True, temperature=temp, num_return_sequences=n)
        output = self.tokenizer.batch_decode(generate_ids[:, len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        cnt = {"OTHER": 0}
        for item in output:
            try:
                word_list = nltk.word_tokenize(item)
                first_word = [word for word in word_list if word.isalnum()][0]
                root_word = get_root_word(first_word)
                if root_word in cnt:
                    cnt[root_word] += 1
                else:
                    cnt[root_word] = 1
            except:
                cnt["OTHER"] += 1
        cnt
        return cnt


class GPTJEstimator:

    def __init__(self, model_name_or_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, pad_token_id=self.tokenizer.eos_token_id).to(device)

    def estimate(self, context, n, temp, max_token):
        input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=max_token, do_sample=True, temperature=temp, num_return_sequences=n)
        output = self.tokenizer.batch_decode(generate_ids[:, len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        cnt = {"OTHER": 0}
        for item in output:
            try:
                word_list = nltk.word_tokenize(item)
                first_word = [word for word in word_list if word.isalnum()][0]
                root_word = get_root_word(first_word)
                if root_word in cnt:
                    cnt[root_word] += 1
                else:
                    cnt[root_word] = 1
            except:
                cnt["OTHER"] += 1
        cnt
        return cnt


class LLaMA2Estimator:

    def __init__(self, model_name_or_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_safetensors=False).half().to(device)

    def get_prompt(self, message, chat_history, system_prompt):
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n[/INST]\n']
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message}')
        return ''.join(texts)

    def estimate(self, context, n, temp, max_token):
        system_prompt = "Please continue writing the following text in English, starting from the next word and not repeating existing content."
        prompt = self.get_prompt(context, [], system_prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=max_token, do_sample=True, temperature=temp, num_return_sequences=n)
        output = self.tokenizer.batch_decode(generate_ids[:, len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        cnt = {"OTHER": 0}
        for item in output:
            try:
                word_list = nltk.word_tokenize(item)
                first_word = [word for word in word_list if word.isalnum()][0]
                root_word = get_root_word(first_word)
                if root_word in cnt:
                    cnt[root_word] += 1
                else:
                    cnt[root_word] = 1
            except:
                cnt["OTHER"] += 1
        cnt
        return cnt


class AlpacaEstimator:

    def __init__(self, model_name_or_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

    def get_prompt(self, instruction, input):
        return f'### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n'

    def estimate(self, context, n, temp, max_token):
        instruction = "Please continue writing the following text in English, starting from the next word and not repeating existing content."
        prompt = self.get_prompt(instruction, context)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=max_token, do_sample=True, temperature=temp, num_return_sequences=n)
        output = self.tokenizer.batch_decode(generate_ids[:, len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        cnt = {"OTHER": 0}
        for item in output:
            try:
                word_list = nltk.word_tokenize(item)
                first_word = [word for word in word_list if word.isalnum()][0]
                root_word = get_root_word(first_word)
                if root_word in cnt:
                    cnt[root_word] += 1
                else:
                    cnt[root_word] = 1
            except:
                cnt["OTHER"] += 1
        cnt
        return cnt


class VicunaEstimator:

    def __init__(self, model_name_or_path, device):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).half().to(device)

    def get_prompt(self, message):
        conv = get_conversation_template(self.model_name_or_path)
        conv.append_message(conv.roles[0], message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt

    def estimate(self, context, n, temp, max_token):
        instruction = "Please continue writing the following text in English, starting from the next word and not repeating existing content: "
        prompt = self.get_prompt(instruction + context)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=max_token, do_sample=True, temperature=temp, num_return_sequences=n)
        output = self.tokenizer.batch_decode(generate_ids[:, len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        cnt = {"OTHER": 0}
        for item in output:
            try:
                word_list = nltk.word_tokenize(item)
                first_word = [word for word in word_list if word.isalnum()][0]
                root_word = get_root_word(first_word)
                if root_word in cnt:
                    cnt[root_word] += 1
                else:
                    cnt[root_word] = 1
            except:
                cnt["OTHER"] += 1
        cnt
        return cnt


class GPT35TurboEstimator:

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def estimate(self, context, n, temp, max_token):
        max_try = 5
        cnt = {"OTHER": 0}
        remaining = n
        while remaining > 0:
            for _ in range(max_try):
                try:
                    completion = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Please continue writing the following text in English, starting from the next word and not repeating existing content."},
                            {"role": "user", "content": context}
                        ],
                        temperature=temp,
                        max_tokens=max_token,
                        n=min(remaining, 128)
                    )

                    for choice in completion.choices:
                        try:
                            word_list = nltk.word_tokenize(choice.message.content)
                            first_word = [word for word in word_list if word.isalnum()][0]
                            root_word = get_root_word(first_word)
                            if root_word in cnt:
                                cnt[root_word] += 1
                            else:
                                cnt[root_word] = 1
                        except:
                            cnt["OTHER"] += 1
                    remaining -= min(remaining, 128)
                    break
                except Exception as e:
                    print(e)
        return cnt

# 获取文本
class GPT4TurboEstimator:

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def estimate(self, context, n, temp, max_token):
        max_try = 5
        cnt = {"OTHER": 0}
        remaining = n
        while remaining > 0:
            for _ in range(max_try):
                try:
                    completion = self.client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        messages=[
                            {"role": "system", "content": "Please continue writing the following text in English, starting from the next word and not repeating existing content."},
                            {"role": "user", "content": context}
                        ],
                        temperature=temp,
                        max_tokens=max_token,
                        n=min(remaining, 128)
                    )

                    for choice in completion.choices:
                        try:
                            word_list = nltk.word_tokenize(choice.message.content)
                            first_word = [word for word in word_list if word.isalnum()][0]
                            root_word = get_root_word(first_word)
                            if root_word in cnt:
                                cnt[root_word] += 1
                            else:
                                cnt[root_word] = 1
                        except:
                            cnt["OTHER"] += 1
                    remaining -= min(remaining, 128)
                    break
                except Exception as e:
                    print(e)
        return cnt


estimators = [
    GPT2Estimator('gpt2-xl', device=torch.device('cuda:3')),
    GPTJEstimator('EleutherAI/gpt-j-6b', device=torch.device('cuda:3')),
    LLaMA2Estimator('meta-llama/Llama-2-13b-chat-hf', token=os.environ['HF_TOKEN'], device=torch.device('cuda:2')),
    AlpacaEstimator('/path/to/alpaca-7b/', device=torch.device('cuda:1')),
    VicunaEstimator('lmsys/vicuna-13b-v1.5', device=torch.device('cuda:0')),
    GPT35TurboEstimator(os.environ['OPENAI_API_KEY']),
    GPT4TurboEstimator(os.environ['OPENAI_API_KEY'])
]

roberta_tknz = RobertaTokenizer.from_pretrained("get_feature/roberta-base")

label2id = {
    'human': 0,
    'gpt2-xl': 1,
    'gpt-j-6b': 2,
    'Llama-2-13b-chat-hf': 3,
    'vicuna-13b-v1.5': 4,
    'alpaca-7b': 5,
    'gpt-3.5-turbo': 6,
    'gpt-4-1106-preview': 7
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--delta', type=float, default=1.2)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    return parser.parse_args()

# 根据给定的上下文、目标词和多个估计器对象，使用线程池并发执行每个估计器的estimate方法来估计目标词的概率。返回的结果是目标词在每个估计器中的负对数估计值组成的列表。
def get_estimate_prob(context, target_word, n, temp, max_token):
    # 初始化结果列表
    ans = []

    # 获取目标词的根词形式
    target_word = get_root_word(target_word)

    # 使用线程池执行并发任务
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交任务到线程池中，每个estimator对象调用estimate方法进行估计
        futures = [executor.submit(estimator.estimate, context, n, temp, max_token) for estimator in estimators]

        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            # 获取每个估计器的结果
            cnt = future.result()

            # 计算目标词的估计概率对数负值（信息熵）
            if target_word in cnt:
                ans.append(-1 * math.log(cnt[target_word] / sum(cnt.values())))
            else:
                ans.append(-1 * math.log(1 / n))

    # 返回结果列表
    return ans

# 对输入的文本进行处理，使用NLTK进行分词，然后根据特定条件将不符合要求的词替换为"[SKIP]"，并记录每个词在原文本中的起始位置。
def get_word_list(text):
    # 使用NLTK进行文本分词
    word_list = nltk.word_tokenize(text)

    # 遍历词列表，处理长度小于2且不是字母或数字的词，将其替换为"[SKIP]"
    for i, word in enumerate(word_list):
        # 处理长度小于2且不是字母或数字的词
        if len(word) < 2 and word not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            word_list[i] = "[SKIP]"
        # 处理双引号符号"``"和"''"
        if word in ["``", "''"]:
            word_list[i] = "[SKIP]"

    # 初始化词位置列表
    pos = []
    word_start = 0

    # 遍历词列表，记录每个词在原文本中的起始位置
    for word in word_list:
        if word == "[SKIP]":
            # 如果是"[SKIP]"，直接记录当前位置并继续下一个词
            pos.append(word_start)
            continue

        # 找到当前词在原文本中的起始位置
        while text[word_start] != word[0]:
            word_start += 1
        pos.append(word_start)
        # 更新起始位置，加上当前词的长度
        word_start += len(word)

    # 返回处理后的词列表和每个词的起始位置列表
    return word_list, pos

# import jieba
# 中文处理
# def get_word_list(text):
#     # 使用jieba进行中文分词
#     word_list = list(jieba.cut(text))
#
#     # 初始化词位置列表
#     pos = []
#     word_start = 0
#
#     # 遍历词列表，记录每个词在原文本中的起始位置
#     for word in word_list:
#         # 找到当前词在原文本中的起始位置
#         while text[word_start:word_start+len(word)] != word:
#             word_start += 1
#         pos.append(word_start)
#         # 更新起始位置，加上当前词的长度
#         word_start += len(word)
#
#     # 返回处理后的词列表和每个词的起始位置列表
#     return word_list, pos
def calc_max_logprob(n, delta):
    min_p = 1 / (1 + n / (1.96/delta)**2)
    # 这一行计算了最小概率min_p。这个计算基于统计学中的置信区间计算公式。具体来说，这个公式用于估计二项分布的成功概率p的下界。在这里，n是样本量，delta是置信度和置信区间宽度的关系参数。1.96是在95 % 置信水平下的标准正态分布的临界值（Z值），用来计算置信区间。
    max_logp = -math.log(min_p)
    # 这一行计算了最大对数概率max_logp。通过取min_p的负对数，可以得到max_logp。在统计学中，通常使用对数变换来处理概率，因为对数转换可以将概率值从(0, 1)范围映射到(-∞, 0] 范围。
    return max_logp


def main(args):
    max_loss = calc_max_logprob(args.n, args.delta)
    # 调用calc_max_logprob函数计算最大对数损失max_loss，使用了从命令行传入的参数args.n和args.delta。
    enc = tiktoken.get_encoding("cl100k_base")
    # 获取了一个名为enc的编码器对象，这里使用了tiktoken模块中的get_encoding函数，并使用了"cl100k_base"作为参数。
    with open(args.input, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    # 将每行解析为JSON格式的数据。
    for item in tqdm(data):
        item['label_int'] = label2id[item['label']]
        # 将item的'label'标签通过label2id字典映射为整数，并存储在'label_int'键中。
        word_list, pos = get_word_list(item['text'])
        # 使用get_word_list函数处理item['text']，返回词列表word_list和词性列表pos。
        proxy_ll = list(zip(range(len(word_list)), item['proxy_prob'], word_list, pos))
        # 创建一个列表proxy_ll
        skip_prefix = 20
        # 过滤掉前skip_prefix个元素和特定词"[SKIP]"。
        proxy_ll = proxy_ll[skip_prefix:]
        proxy_ll = [item for item in proxy_ll if item[2] != "[SKIP]"]

        proxy_ll = [item for item in proxy_ll if item[1] <= max_loss]
        # 过滤掉代理概率大于max_loss的元素。
        proxy_ll = sorted(proxy_ll, key=lambda x: x[1], reverse=True)[:args.k]
        # 按照代理概率降序和索引升序排序，选取前args.k个元素。
        proxy_ll = sorted(proxy_ll, key=lambda x: x[0])

        est_prob = []
        # 对于proxy_ll中的每个元素，根据文本片段和词来估计概率，存储在est_prob中。
        for proxy_ll_item in proxy_ll:
            # 提取当前元素前的文本片段作为上下文context，长度为proxy_ll_item[3]的位置处。
            context = item['text'][:proxy_ll_item[3]]
            # 使用编码器enc对上下文进行编码和解码，截取最后的20个字符作为最终的上下文。
            context = enc.decode(enc.encode(context)[-20:])
            # 调用get_estimate_prob函数估计当前context中出现proxy_ll_item[2]词的概率，
            # 使用args.n、1.5和2作为参数。
            est_prob.append(get_estimate_prob(context, proxy_ll_item[2], args.n, 1.5, 2))

        # 将est_prob转换为NumPy数组，并进行转置操作。
        est_prob = np.array(est_prob)
        est_prob = est_prob.T

        # 将估计的概率列表est_prob转换为Python列表，并将其存储在item的'est_prob_list'键中。
        item['est_prob_list'] = est_prob.tolist()

        # 构建前缀文本列表，其中每个元素是当前proxy_ll中元素之前的文本片段。
        prefix_text_list = [item['text'][:proxy_ll_item[3]] for proxy_ll_item in proxy_ll]

        # 使用roberta_tknz函数处理prefix_text_list，获取目标索引列表target_roberta_idx_list，
        # 其中每个索引表示相应前缀文本在RoBERTa模型输入中的位置。
        target_roberta_idx_list = [len(ids) - 2 for ids in roberta_tknz(prefix_text_list).input_ids]

        # 将目标索引列表target_roberta_idx_list存储在item的'target_roberta_idx'键中。
        item['target_roberta_idx'] = target_roberta_idx_list

        # 构建目标概率索引列表，其中每个元素是当前proxy_ll中元素的位置。
        item['target_prob_idx'] = [proxy_ll_item[3] for proxy_ll_item in proxy_ll]

        # 删除item中的'proxy_prob'和'source'键。
        del item['proxy_prob']
        del item['source']

        # 将修改后的item以JSON格式写入到args.output指定的输出文件中。
        with open(args.output, 'a') as f:
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
