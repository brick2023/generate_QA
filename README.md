# generate_QA

## 簡介
- 使用 OpenAI 的 GPT-3 產生問答集
- 根據語言判斷檔案的語言，並將其轉換成繁體中文
- 將純文字檔案轉換成問答集，並輸出成 json 檔案


# 使用方法

## 必要設置
- 先將 `key` 更改成自己的 OpenAI API key
- 安裝 `openai`, `opencc-python-reimplemented`, `json`, `langdetect` 套件
```bash
pip install openai opencc-python-reimplemented json langdetect
```
- 可以將 GPT-3 的 `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty` 調整成自己想要的數值

## 參數
- data_path : 純文字檔案資料夾路徑
- output_path : 輸出的問答集 json 路徑
- model : 使用的 GPT 模型
