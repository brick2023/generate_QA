# use open ai api to generate text
"""
回傳格式
{
    "id": "chatcmpl-996Wln9ibpN26IVBpqVGJPHqGdbFG",
    "object": "chat.completion",
    "created": 1711957071,
    "model": "gpt-3.5-turbo-0125",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "我是一個聊天機器人，可以回答問題、提供資訊，和用戶進行對話。有什麼可以幫助您的嗎？"
            },
            "logprobs": null,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 11,
        "completion_tokens": 57,
        "total_tokens": 68
    },
    "system_fingerprint": "fp_3bc1b5746c"
}
"""

"""
ChatCompletion(id='chatcmpl-996yAm2rp36Tv9SLDIwYGepjmjSZ9', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='{\n    "instruction": "什麼是生物演化的證據？",\n    "input": "",\n    "output": "生物演化的證據包括化石和生痕化石。化石是生物死後形成的遺骸或遺體，在岩層中保存下來，可以透過化石研究了解古代生物的外貌和演化過程。生痕化石則是古代生物曾經活動過留下的痕跡，例如足跡、巢穴、糞便等，也能提供演化的證據。"\n},\n{\n    "instruction": "生物演化的過程中會有哪些改變？",\n    "input": "",\n    "output": "生物演化的過程中會有體型、腳趾結構、牙齒齒冠等相應的改變。舉例來說，在馬的演化過程中，從古代到現代，馬的大小、前肢腳趾數量、牙齒齒冠的高低都有明顯的變化。"\n},\n{\n    "instruction": "為什麼化石中常見的是生物的骨骼和牙齒？",\n    "input": "",\n    "output": "化石中常見的是生物的骨骼和牙齒，因為這些部位是比較堅硬的組織，容易被保留在岩層中，而不容易腐朽。相較之下，較軟的組織如肌肉和蛋白質易被細菌分解，不容易形成化石。"\n},\n{\n    "instruction": "什麼是生痕化石？",\n    "input": "",\n    "output": "生痕化石是指古代生物曾經活動過留下的痕跡，例如足跡、巢穴、糞便等，而非直接的生物遺骸。透過生痕化石的研究，我們可以了解古代生物的活動模式和環境。"\n},\n{\n    "instruction": "請解釋現代馬和過去馬間的演化變化。",\n    "input": "",\n    "output": "從化石記錄中可以看出，過去馬的前肢腳趾是四指，體型比較小，牙齒齒冠較低，被認為生活在森林中吃樹的嫩葉。隨著時間的推移，馬的演化過程中前肢腳趾逐漸退化為單指，體型變大，牙齒齒冠變高，顯示牠們演化為在平坦草原上生活，食草纖維較多的環境。"\n}', role='assistant', function_call=None, tool_calls=None))], created=1711958770, model='gpt-3.5-turbo-0125', object='chat.completion', system_fingerprint='fp_3bc1b5746c', usage=CompletionUsage(completion_tokens=902, prompt_tokens=7991, total_tokens=8893))
"""

from openai import OpenAI
import json, os
from opencc import OpenCC
from langdetect import detect
import tiktoken

key = ""

client = OpenAI(api_key=key)

def generate_QA(data_path: str = "/home/brick2/plain_text/國中生物大雜燴黑板講解", output_path: str = "output.json", client=client, model="gpt-3.5-turbo", summary_path=None):
    """
    讀取資料夾中的所有檔案，使用 OpenAI API 生成問答集
    集合成一個 json 檔
    data_path: str, 資料夾路徑
    output_path: str, 輸出 json 檔案路徑
    """
    
    def get_completion_json(prompt, model="gpt-3.5-turbo", try_times=4):
        messages = [{"role": "user", "content": prompt}]
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model
            )
        except Exception as e: # 如果發生錯誤，像是重複的 prompt，就跳過這筆資料
            print("Error:", e)
            print("跳過此筆資料")
            return None
        output = chat_completion.choices[0].message.content
        print("output:", output)
        # 檢查回傳值是否為 JSON 格式，若不是則再嘗試一次，最多嘗試 4 次，遞迴呼叫
        try:
            json_output = json.loads(output)
        except:
            if try_times > 0:
                return get_completion_json(prompt, model, try_times-1)
            else:
                return None
        # 並且要檢查格式為 [{instruction: str, input: str, output: str}, ...]
        try:
            if not isinstance(json_output, list):
                return get_completion_json(prompt, model, try_times-1)
            for qa in json_output:
                if not isinstance(qa, dict):
                    return get_completion_json(prompt, model, try_times-1)
                if "instruction" not in qa or "input" not in qa or "output" not in qa:
                    return get_completion_json(prompt, model, try_times-1)
                if not isinstance(qa["instruction"], str) or not isinstance(qa["input"], str) or not isinstance(qa["output"], str):
                    return get_completion_json(prompt, model, try_times-1)
        except:
            return None

        return json_output

    # 讀取文本資料，走訪資料夾
    json_data_list = []
    for file in os.listdir(data_path):
        if not file.endswith(".txt"):
            continue
        text_data = ""
        with open(os.path.join(data_path, file), "r") as f:
            text_data = f.read()

        # 計算 token 數量
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(text_data))
        print("num_tokens:", num_tokens)
        if num_tokens > 4096:
            print("超過 4096 token，使用 summary 來生成問答集")
            if summary_path is None:
                print("summary_path is None")
                continue
            # 到 summary_path 中找相同檔名的檔案
            summary_file_path = os.path.join(summary_path, file)
            if not os.path.exists(summary_file_path):
                print("summary_file_path not exists")
                continue
            with open(summary_file_path, "r") as f:
                text_data = f.read()

        # 簡體轉繁體
        cc = OpenCC("s2twp")
        text_data = cc.convert(text_data)

        # 準備 prompt
        prompt = f"""\"\"\"
        {text_data}
        \"\"\"

        根據以上資料，生成相關的繁體中文問答集，產生約 10 筆資料, JSON 資料格式如下, instruction 為提問的內容, output 為回覆的內容, input 則保留空白：
        [{{
            "instruction": "提問",
            "input": "",
            "output": "回覆"
        }},]"""

        # 生成問答集
        json_content = get_completion_json(prompt)
        if json_content is None:
            continue

        # 將 key 非 instruction, input, output 的項目從 json_content 移除
        for qa in json_content:
            for key in list(qa.keys()):
                if key not in ["instruction", "input", "output"]:
                    print("remove qa:", qa)
                    json_content.remove(qa)
        
        # 檢測回答是否為繁體中文，並移除不符合條件的回答
        for qa in json_content:
            try:
                if detect(qa["output"]) == "en":
                    # 移除不是繁體中文的回答
                    json_content.remove(qa)
                    continue
            except Exception as e:
                print("detect error:", e)
                print("detect 發生問題，跳過檢查")
            if qa["instruction"] == "" or qa["instruction"] == "提問":
                # 移除 instruction 為空的回答
                json_content.remove(qa)
                continue
            if qa["output"] == "" or qa["output"] == "回覆":
                # 移除 output 為空的回答
                json_content.remove(qa)
                continue


        json_data_list.extend(json_content)
        print(json_data_list)
        print("目前資料筆數:", len(json_data_list))

        # 將問答集存成 json 檔
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data_list, f, ensure_ascii=False, indent=4)

# generate_QA(data_path="/home/brick2/plain_text/國中生物大雜燴黑板講解", output_path="gpt-generate-dataset-國中生物大雜燴黑板講解2.json")
if __name__ == "__main__":
    generate_QA(data_path="/home/brick2/plain_text/1_高一生物", output_path="gpt-generate-dataset-高一生物.json")
