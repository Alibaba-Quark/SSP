import json

import requests

# change for your host
VLLM_HOST = "http://localhost:5000"
url = f"{VLLM_HOST}/v1/completions"

headers = {"Content-Type": "application/json"}
data = {
    "prompt": "What is JupySQL?",
    "max_tokens": 100,
    "temperature": 0,
}
models = requests.get(f"{VLLM_HOST}/v1/models", headers=headers).json()["data"]
print(models)
response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json()["choices"][0]["text"])
