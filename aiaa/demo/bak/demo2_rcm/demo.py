# -*- coding: utf-8 -*-
# coding: utf-8

import json
template2 = json.loads("""
{
    "messages": [
        {
            "role": "system",
            "content": "你是一个音乐爱好者，请根据下面描述推荐搜索歌曲用的搜索词，用json数组返回，数组中每一项是一个字符串:"
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

""");


template = json.loads("""{
	"model": "llama-7b",
	"prompt": ["你是一个音乐爱好者，请根据下面描述推荐搜索歌曲用的搜索短语，描述内容中越前面的越重要，把返回的搜索短语格式化成这个格式<!<...>!>，内容尽量200个字以内。"],
	"max_tokens": 200,
	"temperature": 0.7,
	"top_p": 0.9
}""");


import jieba

import aiohttp
import aiohttp.web
import asyncio
import os
import requests
import copy

def searchwords(question):
	target_url = "http://127.0.0.1:8080/v1/completions"
	question_for_search = copy.deepcopy(template);

	question_for_search["prompt"] = "你是一个音乐爱好者\n请推荐在音乐app中搜索歌曲用的搜索词，用<!搜索词1 搜索词2 ...!>格式，根据提问内容中的心情，节日，语言，歌手等信息提取两三个五个字以内的搜索词，内容尽量200个字以内:\n" + question;
	body = json.dumps(question_for_search);
	r = requests.post(target_url,data=body);
	ret = json.loads(r.text)["content"];
	print("搜索词返回:<%s>"%ret);
	pos = ret.find("<!");
	if pos != -1:
		ret1 = ret[pos+2:]
		pos = ret1.find("!>");
		if pos != -1:
			ret = ret1[:pos];
	print("模型给出的搜索短语是:<%s>"%ret);
	return ret;



def search_songs(keyword):
	target_url = "url省略"
	r = requests.post(target_url,params={"keyword":keyword},data='',headers={"X-Appid":"1"});
	songlist = json.loads(r.text)["data"]["songs"]
	return songlist;


async def handle_completions(request):
	# 目标服务器的URL
	target_url = "http://127.0.0.1:8080/v1/chat/completions"


	async def get_prompt_and_request_data():
		txt = await request.content.read()
		obj = json.loads(txt);

		question = obj["messages"][-1]["content"];
		keywords = searchwords(question);
		print("提问内容",question,"关键词是",json.dumps(keywords,indent=4));

		if type(keywords) is str and len(keywords) > 0:
			songlist = search_songs(keywords);
		else:
			songlist = [];


		prompt = "你是一个音乐爱好者，已知歌曲:\n"
		for song in songlist:
			#prompt += "\tkey=" + song["key"] + " name=" + song["name"] + " artist=" + song["artistName"] + " genre=" + song["genreName"] + " url=" + song["linkShare"] + " image=" + song["image"] + "\n"
			prompt += "\tname=" + song["name"] + " artist=" + song["artistName"] + " genre=" + song["genreName"] + " sharelink=" + song["linkShare"] + " image=" + song["image"] + "\n"
		prompt += "请根据以上内容给出推荐结果，每首歌曲单独一行，不要把sharelink做成超链接，推荐结果用`` name | sharelink ``的格式返回，不要推荐列表之外的歌曲，不超过800字:"

		for msg in obj["messages"]:
			if msg["role"] == "system":
				msg["content"] = prompt;
				# print(prompt);
		print("系统提示词",prompt);
		print("输出返回json",json.dumps(obj,indent=4));
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