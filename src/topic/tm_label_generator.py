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
    args.add_argument("--base_path", type=str, default="data/output/") # path to the dataset containing video directories
    args.add_argument("--api_key", type=str, default=None) # API key for the language model
    args.add_argument("--model_name", type=str, default="gemma3:latest") # gemma3:latest  llama3.2:latest

    BASE_DATASET_PATH = args.parse_args().base_path
    model_name = args.parse_args().model_name
    api_key = args.parse_args().api_key

    print(f"Generating Labels using {model_name}")

    # Iterate over all video directories in the dataset
    video_dirs = [d for d in os.listdir(BASE_DATASET_PATH) if os.path.isdir(os.path.join(BASE_DATASET_PATH, d))]

    for video_dir in tqdm(video_dirs, desc="Processing Videos..."):
        video_path = os.path.join(BASE_DATASET_PATH, video_dir)
        topics_info_path = os.path.join(video_path, "topic_info.json")
        
        # Check if topics_info.json exists in this video directory
        if not os.path.exists(topics_info_path):
            print(f"topic_info.json not found in {topics_info_path}. Skipping...")
            continue
        
        data = json.load(open(topics_info_path))
        for topic in tqdm(data, desc=f"Processing Topics in {video_dir}..."):
            if topic['Topic'] == -1:
                continue
            keywords = topic['Representation']
            documents = topic['Representative_Docs']
            topic_prompt = prompt.format(keywords=keywords, documents=documents)
            response = generate_label(model_name, api_key, topic_prompt)
            topic['label'] = response['choices'][0]['message']['content'].split('\n')[0]

        cleaned_topics_info_path = os.path.join(video_path, "cleaned_topics_info.json")
        json.dump(data, open(cleaned_topics_info_path, 'w+'), indent=4)

