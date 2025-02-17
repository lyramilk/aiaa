# -*- coding: utf-8 -*-
# coding: utf-8

import json
import os
import copy
import requests


class ChatDataIter:
	def __init__(self,cd):
		self.index = 0;
		self.data = cd.data;
	
	def __next__(self):
		if self.index >= len(self.data):
			raise StopIteration

		value = self.data[self.index]
		self.index += 1
		return value

class ChatData:
	def __init__(self):
		self.appkey = os.environ.get('DEEPSEEK_APIKEY');
		self.data = []
		self.system_prompt = None;
		
		self.template = {
			"model": "deepseek-chat",
			"messages": [],
			"response_format": {
				"type": "json_object"
			},
			"stream": False
		}
		
	def __getitem__(self,index):
		return self.data.get(index);

	def __setitem__(self,index,value):
		self.data[index] = value;
		return self;

	def __delitem__(self,index):
		return self.data.pop(index);

	def __iter__(self):
		return ChatDataIter(self);

	def clear(self):
		self.data = [];

	def prompt(self,system_prompt):
		self.system_prompt = system_prompt;


	def append(self,chat):
		if len(self.data) == 0:
			self.data.append({"role": "user", "content": chat});
			return;
		lastRole = self.data[-1]["role"];
		
		if lastRole == "assistant":
			self.data.append({"role": "user", "content": chat});
			return;
		if lastRole == "user":
			self.data.append({"role": "assistant", "content": chat});
			return;
		raise Exception("unknow role %s"%lastRole);

	def get(self):
		target_url = "https://api.deepseek.com/v1/chat/completions";
		
		self.template["data"] = self.data;
		
		messages  = [];
		if self.system_prompt:
			messages.append({"role": "system", "content": self.system_prompt});
		messages.extend(self.data);
		
		reqbody = copy.deepcopy(self.template);
		reqbody["messages"] = messages;
		
		r = requests.post(target_url,headers={"Content-Type": "application/json","Authorization": "Bearer " + self.appkey},data=json.dumps(reqbody));
		return json.loads(json.loads(r.text)["choices"][-1]["message"]["content"]);




session = ChatData();
session.prompt("""
已知的函数列表:
LARK(<message>,<level>)  用lark发送报警
IGNORE()  忽略
请分析下面syslog并使用json格式返回
```
{
	"c":<要调用的函数>
	"p":[<参数1>,<参数2>...]
}
```
""");


session.clear();
session.append("""
Jan  2 19:10:01 vn-vdc-192-168-100-133 CRON[1240103]: (root) CMD (/usr/sbin/ntpdate 192.168.100.19 >/dev/null 2>&1)
Jan  2 19:10:01 vn-vdc-192-168-100-133 CRON[1240104]: (lyramilk) CMD (/home/lyramilk/.local/bin/python3-lark /data/lyramilk/ClusterSS/src/lyramilk/cavedb_music_alive.task.py)
Jan  2 19:10:01 vn-vdc-192-168-100-133 CRON[1240099]: (CRON) info (No MTA installed, discarding output)
Jan  2 19:11:01 vn-vdc-192-168-100-133 CRON[1240113]: (lyramilk) CMD (/home/lyramilk/.local/bin/python3-lark /data/lyramilk/ClusterSS/src/lyramilk/cavedb_music_alive.task.py)
Jan  2 19:11:01 vn-vdc-192-168-100-133 CRON[1240112]: (CRON) info (No MTA installed, discarding output)
Jan  2 19:12:01 vn-vdc-192-168-100-133 CRON[1240169]: (lyramilk) CMD (/home/lyramilk/.local/bin/python3-lark /data/lyramilk/ClusterSS/src/lyramilk/cavedb_music_alive.task.py)
Jan  2 19:12:01 vn-vdc-192-168-100-133 CRON[1240168]: (CRON) info (No MTA installed, discarding output)
Jan  2 19:13:01 vn-vdc-192-168-100-133 CRON[1240175]: (lyramilk) CMD (/home/lyramilk/.local/bin/python3-lark /data/lyramilk/ClusterSS/src/lyramilk/cavedb_music_alive.task.py)
Jan  2 19:13:01 vn-vdc-192-168-100-133 CRON[1240174]: (CRON) info (No MTA installed, discarding output)
Jan  2 19:14:01 vn-vdc-192-168-100-133 CRON[1240184]: (lyramilk) CMD (/home/lyramilk/.local/bin/python3-lark /data/lyramilk/ClusterSS/src/lyramilk/cavedb_music_alive.task.py)
Jan  2 19:14:01 vn-vdc-192-168-100-133 CRON[1240183]: (CRON) info (No MTA installed, discarding output)
Jan  2 19:15:01 vn-vdc-192-168-100-133 CRON[1240192]: (root) CMD (command -v debian-sa1 > /dev/null && debian-sa1 1 1)
Jan  2 19:15:01 vn-vdc-192-168-100-133 CRON[1240193]: (lyramilk) CMD (/home/lyramilk/.local/bin/python3-lark /data/lyramilk/ClusterSS/src/lyramilk/cavedb_music_alive.task.py)
""");

r = session.get();

print(json.dumps(r,indent=4));