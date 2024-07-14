from MultiModalDeepFakemain.code import detectmul
from POGERmain.POGER import detect
from SeqXGPT.SeqXGPT import detect2
def detect_aigc_chinese(text):
    """
    检测中文文本是否为AIGC生成，并尝试给出来源（伪代码）
    """
    is_aigc, source = detect2.detect_aigc_chinese(text)
    # 类似地，这里也是伪代码

    return is_aigc, source


def detect_aigc_non_chinese(text):
    is_aigc, source=detect.detect_aigc_non_chinese(text)
    # 类似地，这里也是伪代码

    return is_aigc, source

def detect_aigc_mul(path):
    """
    检测多模态图片是否为AIGC生成
    """
    text,detect_per=detectmul.detect_aigc_mul(path)
    return text,detect_per


def main():
    print("欢迎使用AIGC检测工具")

    while True:
        print("请选择检测的类型：")
        print("1: 多模态检测")
        print("2: 文本检测")
        print("3: 退出")

        choice = input("请输入选项：")

        if choice == '1':
            path = input("请输入上传的图片地址：")
            text, detect_per = detect_aigc_mul(path)
            if detect_per > 50:
                print(f"{text}是AIGC生成的，可能性：{detect_per}%")
            else:
                print(f"{text}不是AIGC生成的")

        elif choice == '2':
            language_choice = input("请选择检测的语言类型（1: 中文，2: 非中文）：")

            if language_choice == '1':
                text = input("请输入中文文本进行检测：")
                is_aigc, source = detect_aigc_chinese(text)
                if is_aigc:
                    print(f"该文本是AIGC生成的，来源：{source}")
                else:
                    print("该文本不是AIGC生成的。")

            elif language_choice == '2':
                text = input("请输入非中文文本进行检测：")
                is_aigc, source = detect_aigc_non_chinese(text)
                if is_aigc:
                    print(f"该文本是AIGC生成的，来源：{source}")
                else:
                    print("该文本不是AIGC生成的。")

            else:
                print("无效的输入，请输入1或2。")

        elif choice == '3':
            print("退出程序。")
            break

        else:
            print("无效的输入，请输入1、2或3。")

if __name__ == "__main__":
    main()