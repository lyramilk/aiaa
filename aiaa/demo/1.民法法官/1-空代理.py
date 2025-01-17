import aiohttp
import aiohttp.web
import asyncio
import os
import json

project_dir = "/data/case/lyramilk/aiaa/aiaa"
dataset_dir = project_dir + "/dataset"
static_dir = project_dir + "/static"
appkey = os.environ.get('DEEPSEEK_APIKEY');


"""
# llama.cpp返回格式的参考，大模型的json格式都是兼容的，llama.cpp的全一点，deepseek官网给的例子有很多字段没列出来。
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

	# 获取上行数据。
	async def get_upward_data():
		upward_data = await request.content.read();
		# 美化一下。顺便解决一点点兼容性问题。
		chatobj = json.loads(upward_data);
		chatobj["model"] = "deepseek-chat"
		if "max_tokens" in chatobj:
			chatobj["max_tokens"] = 4096 if chatobj["max_tokens"] == -1 else chatobj["max_tokens"]
		prettyjson = json.dumps(chatobj,indent=4);
		print(prettyjson);
		return prettyjson;

	# 异步代理:
	# 创建一个指向大模型接口的http请求会话 和 一个响应流对象。大模型接口每返回一块就把这一块写给响应流，简单实现异步的http代理。
	async with aiohttp.ClientSession() as session:
		# 准备转发请求
		async with session.post("https://api.deepseek.com/v1/chat/completions", data=await get_upward_data(),headers={"Content-Type": "application/json","Authorization": "Bearer " + appkey}) as resp:
			if resp.status != 200:
				chunk = await resp.content.read();
				print(chunk);
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


async def handle_chat_completions(request):

	# 获取上行数据。
	async def get_upward_data():
		upward_data = await request.content.read();
		# 美化一下。顺便解决一点点兼容性问题。
		chatobj = json.loads(upward_data);
		chatobj["model"] = "deepseek-chat"
		if "max_tokens" in chatobj:
			chatobj["max_tokens"] = 4096 if chatobj["max_tokens"] == -1 else chatobj["max_tokens"]
		prettyjson = json.dumps(chatobj,indent=4);
		print("请求:",prettyjson);
		return prettyjson,chatobj.get("stream") or False;

	# 异步代理:
	# 创建一个指向大模型接口的http请求会话 和 一个响应流对象。大模型接口每返回一块就把这一块写给响应流，简单实现异步的http代理。
	async with aiohttp.ClientSession() as session:
		# 准备转发请求
		upward_data,isstream = await get_upward_data()
		async with session.post("https://api.deepseek.com/chat/completions", data=upward_data,headers={"Content-Type": "application/json","Authorization": "Bearer " + appkey}) as resp:
			print(resp.status);
			if resp.status != 200:
				chunk = await resp.content.read();
				return aiohttp.web.Response(text="Failed to proxy request", status=resp.status)

			if isstream:
				response = aiohttp.web.StreamResponse(status=resp.status, headers=resp.headers)
				await response.prepare(request)
				while True:
					chunk = await resp.content.read(8192)
					if not chunk:
						break
					print(json.dumps(json.loads(chunk),indent=4));
					await response.write(chunk)
				await response.write_eof()
				return response
			else:
				print("是流吗",isstream);
				return aiohttp.web.Response(body=await resp.content.read(),status=resp.status, headers=resp.headers)

# 主页，聊天页面
async def handle_request(request):
	# 获取聊天页面的文件路径，这个html文件是取的llama.cpp项目的examples/server/public/index.html文件，后面会自己写，现在先用现成的。
	file_path = static_dir + "/llama.cpp.html";
	# 检查文件是否存在
	if not os.path.exists(file_path):
		return aiohttp.web.Response(text="File not found", status=404)
	# 读取文件内容
	with open(file_path, 'rb') as f:
		content = f.read()
	return aiohttp.web.Response(body=content, content_type='text/html')


# 主页，聊天页面
async def handle_any(request):
	print(request.path);
	return aiohttp.web.Response(body="ok", content_type='text/plain')
# 运行代理服务器
if __name__ == '__main__':
	# 创建应用程序对象
	app = aiohttp.web.Application()
	# 添加路由
	app.router.add_route('*','/{url:.*}', handle_any)
	app.router.add_get('/', handle_request)
	app.router.add_post('/v1/chat/completions', handle_completions)
	app.router.add_post('/chat/completions', handle_chat_completions)
	# 启动服务
	aiohttp.web.run_app(app,host='127.0.0.1', port=8888)
