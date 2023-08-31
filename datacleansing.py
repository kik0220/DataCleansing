import os
import json
import gradio as gr

source_file_path = ""

config_json = {}
script_dir = os.getcwd()
user_config_path = os.path.join(script_dir, "user_config.json")
if os.path.exists(user_config_path):
    with open(user_config_path, 'r', encoding='utf-8') as file:
        config_json = json.load(file)

def source_file_open(source_file):
    with open(source_file, 'r', encoding='utf-8') as file:
        source_text = file.read()
    return source_text.split("\n")

def removed_file_save(removed_text):
    file_name_with_ext = os.path.basename(source_file_path)
    file_name, file_extension = os.path.splitext(file_name_with_ext)
    save_file_path = f"{script_dir}/{file_name}_removed{file_extension}"
    with open(save_file_path, 'w', encoding='utf-8') as file:
        file.write(removed_text)
        file.close()

def source_file(source_path,source_display):
    global source_file_path
    lines = source_file_open(source_path.name)
    source_display = [[line] for line in lines]
    source_file_path = source_path.name
    return source_path,source_display

def texts_run(run_button,remove_texts):
    remove_words = remove_texts.split("\n")
    remove_words = [word for word in remove_words if word != '']
    lines = source_file_open(source_file_path)
    filtered_lines = [line for line in lines if all(word not in line for word in remove_words)]
    removed_text =  '\n'.join(filtered_lines)
    removed_file_save(removed_text)
    return run_button

def length_run(run_button,remove_length):
    lines = source_file_open(source_file_path)
    filtered_lines = [line for line in lines if len(line) > int(remove_length)]
    removed_text = '\n'.join(filtered_lines)
    removed_file_save(removed_text)
    return run_button

with gr.Blocks(title="Data cleansing") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            source_path = gr.UploadButton ("ソースファイルを選択", file_types=["text"],container=True)
            remove_texts = gr.TextArea(label="テキストから除外したいワード",placeholder="1行1ワードで入力できます",container=True)
            texts_run_button = gr.Button("実行")
            remove_length = gr.Textbox(label="テキストから除外したい長さ",placeholder="30",container=True)
            length_run_button = gr.Button("実行")
        with gr.Column(scale=10):
            source_display = gr.List(headers=["ソースファイル"],col_count=(1, "fixed"),max_rows=20)

    source_path.upload(source_file,[source_path,source_display],[source_path,source_display])
    texts_run_button.click(texts_run,[texts_run_button,remove_texts],texts_run_button)
    length_run_button.click(length_run,[length_run_button,remove_length],length_run_button)

if __name__ == "__main__":
    if "server_port" in config_json:
        config_server_port = config_json["server_port"]
    else:
        config_server_port = None
    if "server_name" in config_json:
        config_server_name = config_json["server_name"]
    else:
        config_server_name = None
    if "inbrowser" in config_json:
        config_inbrowser = config_json["inbrowser"]
    else:
        config_inbrowser = False
    demo.launch(share=False,show_api=False,server_port=config_server_port,server_name=config_server_name,inbrowser=config_inbrowser)
