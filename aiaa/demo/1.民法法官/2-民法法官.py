# jieba是个分词库 pip3 install jieba
import jieba

project_dir = "/data/case/lyramilk/aiaa/aiaa"
dataset_dir = project_dir + "/dataset"
static_dir = project_dir + "/static"

# 过滤一些没有意义的词。
stopwords = set([
	"的", "是", "在", "了", "和", "或", "因为", "所以", "这", "那", "它", "我们", "你们", "他们",
	"啊", "呀", "呢", "吧", "吗", "哦", "啦", "哇", "唉", "哼",
	"着", "过", "地", "得", "之", "乎", "者", "也",
	"一个","一下", "一些", "几个", "很多", "少量", "大量", "一部分", "大多数", "少数", "全部",
	"，","。","！","？","、","：","“","”","(",")","[","]","{","}",
	",",".","!","?",'"',"'",":","（","）","【","】","{","}",
])
def jieba_cut(txt):
	print(type(txt));
	return [x for x in list(jieba.cut(txt)) if x not in "，。！？、,.!?" and x not in stopwords]



datasource = [];
# 构建搜索索引
with open(dataset_dir + "/民法典.txt","r") as f:
	datasource = [x.strip() for x in f.read().split("\n")];

# 从上面数据源中找到所有包含某个词的行。
def find_word(ds,word):
	result = [];
	for line in ds:
		if word in line:
			result.append(line);
	return result;

# 从数据源中找到包含指定词组中任意词的行，并记录每一个找到的行被引用的次数，按引用次数从高到低排序，然后取被引用次数最多的前count个行
def find_keywords(ds,keywords,count):
	line_rc = {};

	for word in keywords:
		for line in find_word(ds,word):
			if line not in line_rc:
				line_rc[line] = 1;
			else:
				line_rc[line] += 1;
	

	reference_list = list(line_rc.items());
	reference_list.sort(key=lambda x:x[1],reverse=True);
	return [ x[0] for x in reference_list[:count]];

import aiohttp
import aiohttp.web
import asyncio
import os
import json

project_dir = "/data/case/lyramilk/aiaa/aiaa"
dataset_dir = project_dir + "/dataset"
static_dir = project_dir + "/static"
appkey = os.environ.get('DEEPSEEK_APIKEY');

async def handle_completions(request):

	# 获取上行数据。
	async def get_upward_data():
		upward_data = await request.content.read();
		# 美化一下。顺便解决一点点兼容性问题。
		chatobj = json.loads(upward_data);
		chatobj["model"] = "deepseek-chat"
		chatobj["max_tokens"] = 4096 if chatobj["max_tokens"] == -1 else chatobj["max_tokens"]

		# 取列表最后一项(也就是用户最新输入的案情信息)
		last = chatobj["messages"][-1]["content"];
		# 对案情信息进行分词并搜索前20条被引用次数最多的民法典条款
		keywords = jieba_cut(last);
		references20 = find_keywords(datasource,keywords,20);

		# 根据民法典替换系统提示词
		prompt = "你是一个法官，与法官角色无关的问题全都回答不知道，已知民法条款:\n"
		for referenceline in references20:
			prompt += "\t" + referenceline + "\n"
		prompt += "请根据以上法律条文给出审判结果，说明引用了哪一条，不要引用其它法律条文:"
		# 替换掉系统提示此，这个messages列表第一项就是系统提示词，更严谨的做法查找 role == system的那一项并替换它的content
		print("系统提示词:",prompt);
		chatobj["messages"][0]["content"] = prompt;

		prettyjson = json.dumps(chatobj,indent=4);
		return prettyjson;

	# 异步代理:
	# 创建一个指向大模型接口的http请求会话 和 一个响应流对象。大模型接口每返回一块就把这一块写给响应流，简单实现异步的http代理。
	async with aiohttp.ClientSession() as session:
		# 准备转发请求
		async with session.post("https://api.deepseek.com/v1/chat/completions", data=await get_upward_data(),headers={"Content-Type": "application/json","Authorization": "Bearer " + appkey}) as resp:
			if resp.status != 200:
				chunk = await resp.content.read();
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

# 运行代理服务器
if __name__ == '__main__':
	# 创建应用程序对象
	app = aiohttp.web.Application()
	# 添加路由
	app.router.add_get('/', handle_request)
	app.router.add_post('/v1/chat/completions', handle_completions)
	# 启动服务
	aiohttp.web.run_app(app,host='127.0.0.1', port=8888)
