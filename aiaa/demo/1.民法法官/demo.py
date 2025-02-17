import jieba

def readdata():
	with open("/data/ai/dataset/民法典.txt","r") as f:
		txt = f.read();
		return txt;
txt = readdata().split("\n");


def findSlice(txt,w):
	rs = [];
	for k in txt:
		if w in k:
			rs.append(k);
	return rs;


def pick(content):
	r = jieba.cut(content);
	s = set(list(r));
	if "\n" in s:
		s.remove("\n");

	dt = {};

	for w in s:
		rr = findSlice(txt,w);
		for r in rr:
			if r not in dt:
				dt[r] = 0;
			dt[r] += 1;

	v = list(dt.items());
	v.sort(key=lambda x:x[1],reverse=True);
	return v[:20];
#data = pick(ask2);

import aiohttp
import aiohttp.web
import asyncio
import os
import json
"""
{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "id": 1734860742156,
            "role": "user",
            "content": "\u6253\u4eba"
        }
    ],
    "stream": true,
    "cache_prompt": true,
    "samplers": "dkypmxt",
    "temperature": 0.8,
    "dynatemp_range": 0,
    "dynatemp_exponent": 1,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "typical_p": 1,
    "xtc_probability": 0,
    "xtc_threshold": 0.1,
    "repeat_last_n": 64,
    "repeat_penalty": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "dry_multiplier": 0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,
    "max_tokens": -1,
    "timings_per_token": false
}

"""
async def handle_completions(request):
	# 目标服务器的URL
	target_url = "http://127.0.0.1:8080/v1/chat/completions"

	async def get_prompt_and_request_data():
		txt = await request.content.read()
		obj = json.loads(txt);

		last = obj["messages"][-1];
		data = pick(last["content"]);
		prompt = "你是一个法官，已知民法条款:\n"
		for t in data:
			prompt += "\t" + t[0] + "\n"
		prompt += "请根据以上法律条文给出审判结果，不要引用其它法律条文:"

		for msg in obj["messages"]:
			if msg["role"] == "system":
				msg["content"] = prompt;
				# print(prompt);
		#print(json.dumps(obj,indent=4));
		return json.dumps(obj);
	# 创建一个会话对象
	async with aiohttp.ClientSession() as session:
		# 准备转发请求
		async with session.post(target_url, data=await get_prompt_and_request_data()) as resp:

			if resp.status != 200:
				return aiohttp.web.Response(text="Failed to proxy request", status=resp.status)
			response = aiohttp.web.StreamResponse(status=resp.status, headers=resp.headers)
			await response.prepare(request)
			while True:
				chunk = await resp.content.read(8192)
				if not chunk:
					break
				print(chunk);
				await response.write(chunk)
			await response.write_eof()
			return response

async def handle_request(request):
	# 获取请求的路径
	file_path = "static/index.html";
	# 检查文件是否存在
	if not os.path.exists(file_path):
		return aiohttp.web.Response(text="File not found", status=404)
	# 读取文件内容
	with open(file_path, 'rb') as f:
		content = f.read()
	return aiohttp.web.Response(body=content, content_type='text/html')

async def start_proxy():
	# 创建应用程序对象
	app = aiohttp.web.Application()
	# 添加路由
	app.router.add_get('/', handle_request)
	app.router.add_post('/v1/chat/completions', handle_completions)
	return app;

# 运行代理服务器
if __name__ == '__main__':
	aiohttp.web.run_app(start_proxy(),host='127.0.0.1', port=8888)
	#asyncio.run(start_proxy());