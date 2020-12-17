import requests
import torch
import time

# print('Version', torch.__version__)
# print('CUDA enabled:', torch.cuda.is_available())

start = time.time()

data = {"seedWords": "hello, I am daniel ", "model": "office"}
res = requests.post("http://127.0.0.1:5000/generateText", data=data)
print(res.json()["text"])


data = {"text": res.json()['text'], "model": "office"}
res = requests.post("http://127.0.0.1:5000/generateCellVis", data=data)
print(res.text)

end = time.time()
print(end - start)





