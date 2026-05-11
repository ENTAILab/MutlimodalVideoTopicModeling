import argparse
import json
import os


import requests
from tqdm import tqdm


def generate_label(model_name, token, user_input):
    url = 'https://llm.texttechnologylab.org/api/chat/completions'
    if token is None:
        LookupError(f"API Token not provided for {model_name}. Use --api_key when running script.")
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

prompt = """
You are given a topic that contains a set of representative text excerpts from TV News Broadcast in German. Each topic is also described by a set of keywords. Your goal is to generate a **short, descriptive label** for the topic that accurately reflects the main issue or campaign topic being discussed. Focus first on the keywords, but also look at the content of the representative documents to understand the context.".

Topic information:

Keywords: {keywords}

Top 3 representative documents:
{documents}

Please provide **only the label**, no extra text or explanation.

"""


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--base_path", type=str, required=True)
    args.add_argument("--api_key", type=str, required=True)
    args.add_argument("--model_name", type=str, default="gemma3:latest") # gemma3:latest  llama3.2:latest

    BASE_TM_RESULTS_PATH = args.parse_args().base_path
    model_name = args.parse_args().model_name
    api_key = args.parse_args().api_key

    print(f"Generating Labels using {model_name}")

    tm_result_files = [filename for filename in os.listdir(BASE_TM_RESULTS_PATH) if filename.endswith('.json')]

    for filename in tqdm(tm_result_files, desc=f"Processing Files..."):
        filepath = os.path.join(BASE_TM_RESULTS_PATH,filename)
        data = json.load(open(filepath))
        for topic in tqdm(data, desc=f"Processing Topics..."):
            if topic['Topic'] == -1:
                continue
            keywords = topic['Representation']
            documents = topic['Representative_Docs']
            topic_prompt = prompt.format(keywords=keywords, documents=documents)
            response = generate_label(model_name, api_key, topic_prompt)
            topic['label'] = response['choices'][0]['message']['content'].split('\n')[0]


        json.dump(data, open(os.path.join(BASE_TM_RESULTS_PATH, filename), 'w+'), indent=4)

