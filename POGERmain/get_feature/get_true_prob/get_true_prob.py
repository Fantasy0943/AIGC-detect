import httpx
import msgpack
import threading
import json
from tqdm import tqdm


def access_api(text, api_url, do_generate=False):
    """
    访问给定的API URL，向其发送POST请求，传递文本数据和是否生成的标志，并返回响应内容。

    Args:
    text (str): 输入的文本数据。
    api_url (str): API的URL地址。
    do_generate (bool, optional): 是否执行生成操作的标志，默认为False。

    Returns:
    dict or None: API的响应内容，解码为字典形式。如果请求失败，则返回None。
    """

    # 使用httpx库创建一个客户端对象，设置超时时间为无限
    with httpx.Client(timeout=None) as client:
        # 准备POST请求的数据，包括文本内容和是否生成的标志
        post_data = {
            "text": text,
            "do_generate": do_generate,
        }

        # 发送POST请求到指定的API URL，使用msgpack打包数据
        prediction = client.post(api_url,
                                 data=msgpack.packb(post_data),
                                 timeout=None)

    # 检查响应的状态码，如果是200，则解码响应内容为字典形式返回
    if prediction.status_code == 200:
        content = msgpack.unpackb(prediction.content)
    else:
        # 如果请求失败，将内容设为None
        content = None

    return content

def get_features(input_file, output_file):
    """
    从原始行中获取 [losses, begin_idx_list, ll_tokens_list, label_int, label]。

    Args:
    input_file (str): 输入文件路径，包含原始数据行。
    output_file (str): 输出文件路径，将处理后的结果写入其中。

    Returns:
    None
    """

    # 定义多个模型的推理API地址
    gpt_2_api = 'http://127.0.0.1:6001/inference'
    gpt_J_api = 'http://127.0.0.1:6002/inference'
    llama_2_api = 'http://127.0.0.1:6003/inference'
    alpaca_api = 'http://127.0.0.1:6004/inference'
    vicuna_api = 'http://127.0.0.1:6005/inference'

    # 将模型API地址存放在列表中
    model_apis = [gpt_2_api, gpt_J_api, llama_2_api, alpaca_api, vicuna_api]

    # 定义标签映射关系
    labels = {
        'human': 0,
        'gpt2-xl': 1,
        'gpt-j-6b': 2,
        'Llama-2-13b-chat-hf': 3,
        'vicuna-13b-v1.5': 4,
        'alpaca-7b': 5,
        'gpt-3.5-turbo': 6,
        'gpt-4-1106-preview': 7
    }

    # 打开输入文件，读取每一行的JSON数据
    with open(input_file, 'r') as f:
        lines = [json.loads(line) for line in f]

    # 打印输入文件信息
    print('input file:{}, length:{}'.format(input_file, len(lines)))

    # 打开输出文件，准备写入处理后的结果
    with open(output_file, 'w', encoding='utf-8') as f:
        # 遍历每一行数据
        for data in tqdm(lines):
            # 获取文本内容和标签
            line = data['text']
            label = data['label']

            # 初始化空列表，用于存储每个模型的损失、起始词索引、生成的token列表
            losses = []
            begin_idx_list = []
            ll_tokens_list = []

            # 获取标签的整数表示
            label_int = labels[label]

            # 标记是否发生错误
            error_flag = False

            # 遍历每个模型的API地址
            for api in model_apis:
                try:
                    # 调用访问API函数，获取损失、起始词索引、生成的token列表
                    loss, begin_word_idx, ll_tokens = access_api(line, api)
                except TypeError:
                    # 捕获类型错误，可能是由于GPU内存不足导致的异常
                    print("return NoneType, probably gpu OOM, discard this sample")
                    error_flag = True
                    break

                # 将获取的结果存入对应的列表中
                losses.append(loss)
                begin_idx_list.append(begin_word_idx)
                ll_tokens_list.append(ll_tokens)

            # 如果发生错误，跳过当前样本的处理
            if error_flag:
                continue

            # 构建结果字典，包括损失、起始词索引、生成的token列表、标签整数表示、原始文本和标签
            result = {
                'losses': losses,
                'begin_idx_list': begin_idx_list,
                'll_tokens_list': ll_tokens_list,
                'label_int': label_int,
                'label': label,
                'text': line
            }

            # 将结果字典以JSON格式写入输出文件
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    input_files = ['../../data/train.jsonl',
                   '../../data/val.jsonl',
                   '../../data/test.jsonl']

    output_files = ['./result/train_true_prob.jsonl',
                   './result/val_true_prob.jsonl',
                   './result/test_true_prob.jsonl']

    threads = []
    # 循环遍历input_files列表的每个索引
    for i in range(len(input_files)):
        # 创建一个新的线程，目标函数是get_features，传入参数是input_files[i]和output_files[i]
        t = threading.Thread(target=get_features, args=(input_files[i], output_files[i]))

        # 将新创建的线程对象加入到线程列表threads中
        threads.append(t)

        # 启动线程，开始执行get_features函数
        t.start()

    # 循环遍历线程列表threads中的每个线程对象t
    for t in threads:
        # 等待线程t执行完毕
        t.join()
