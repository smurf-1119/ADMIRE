import os
import re
import pandas as pd

path = '/mntnlp/zqp/physical_report/Qwen2-VL/logs/notrain/'


def get_information(text):
    # Regular expressions to match the required values
    avg_token_len_pattern = r"avg vtokens tensor\((\d+\.\d+)\)"
    avg_total_token_len_pattern = r"avg total tokens (\d+\.\d+)"
    # avg_first_token_latency_pattern = r"avg first token latency: (\d+\.\d+)"
    overall_anls_pattern = r"Overall ANLS: (\d+\.\d+)"

    # Finding matches
    avg_token_len = float(re.search(avg_token_len_pattern, text).group(1))
    avg_total_token_len = float(re.search(avg_total_token_len_pattern, text).group(1))
    # avg_first_token_latency = float(re.search(avg_first_token_latency_pattern, text).group(1))
    overall_anls = [float(anls) * 100 for anls in re.findall(overall_anls_pattern, text)]

    # return avg_token_len, avg_total_token_len, avg_first_token_latency, overall_anls
    return avg_token_len, avg_total_token_len, overall_anls


datas = {}
for name in os.listdir(path):
    filename = os.path.join(path, name)
    text = open(filename).read()
    
    try:
        avg_token_len, avg_total_token_len, overall_anls = get_information(text)
        if len(overall_anls) == 1:
            datas[name.replace('.log', '')]={
                    'vtoken len': avg_token_len,
                    'total token len': avg_total_token_len,
                    # 'ftls': avg_first_token_latency,
                    'recall': '',
                    'overall anls': overall_anls[0],
                    '[1-5]': None,
                    '[5-10]': None,
                    '[10-15]': None,
                    '[15-20]':None,
                    '[20-]': None,
                }
        else:
            datas[name.replace('.log', '')]={
                    'vtoken len': avg_token_len,
                    'total token len': avg_total_token_len,
                    # 'ftls': avg_first_token_latency,
                    'recall': '',
                    'overall anls': overall_anls[0],
                    '[1-5]': overall_anls[1],
                    '[5-10]': overall_anls[2],
                    '[10-15]': overall_anls[3],
                    '[15-20]': overall_anls[4],
                    '[20-]': overall_anls[5],
                }
    except Exception as e:
        print(str(e), name)
        datas[name.replace('.log', '')]={
                    'vtoken len': '',
                    'total token len': '',
                    # 'ftls': '',
                    'recall': '',
                    'overall anls': '',
                    '[1-5]': '',
                    '[5-10]': '',
                    '[10-15]': '',
                    '[15-20]':'',
                    '[20-]': '',
                }


pd.DataFrame(datas).T.to_csv(os.path.join(path, 'notrain_qwen2vl-7b.csv'))