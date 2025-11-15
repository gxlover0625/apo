"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import time
import requests
import config
import string
import os
from openai import OpenAI

def parse_sectioned_prompt(s):

    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result


def chatgpt(prompt, temperature=0.7, n=1, top_p=1, stop=None, max_tokens=4096, 
                  presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=10):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"]
    )

    messages = [{"role": "user", "content": prompt}]
    
    retries = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=os.environ['OPENAI_MODEL'],
                messages=messages,
                temperature=temperature,
                n=n,
                max_tokens=max_tokens
            )
            break
        except Exception as e:
            print(f"调用openai接口出错: {e}，重试")
            retries += 1
            time.sleep(5)
            if retries > 10:
                raise

    # payload = {
    #     "messages": messages,
    #     "model": "gpt-3.5-turbo",
    #     "temperature": temperature,
    #     "n": n,
    #     "top_p": top_p,
    #     "stop": stop,
    #     "max_tokens": max_tokens,
    #     "presence_penalty": presence_penalty,
    #     "frequency_penalty": frequency_penalty,
    #     "logit_bias": logit_bias
    # }
    # retries = 0
    # while True:
    #     try:
    #         r = requests.post('https://api.openai.com/v1/chat/completions',
    #             headers = {
    #                 "Authorization": f"Bearer {config.OPENAI_KEY}",
    #                 "Content-Type": "application/json"
    #             },
    #             json = payload,
    #             timeout=timeout
    #         )
    #         if r.status_code != 200:
    #             retries += 1
    #             time.sleep(1)
    #         else:
    #             break
    #     except requests.exceptions.ReadTimeout:
    #         time.sleep(1)
    #         retries += 1
    # r = r.json()
    # return [choice['message']['content'] for choice in r['choices']]
    return [choice.message.content for choice in response.choices]


def instructGPT_logprobs(prompt, temperature=0.7):
    payload = {
        "prompt": prompt,
        "model": "text-davinci-003",
        "temperature": temperature,
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True
    }
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/completions',
                headers = {
                    "Authorization": f"Bearer {config.OPENAI_KEY}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=10
            )  
            if r.status_code != 200:
                time.sleep(2)
                retries += 1
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(5)
    r = r.json()
    return r['choices']


