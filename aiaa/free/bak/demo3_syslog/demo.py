# -*- coding: utf-8 -*-
# coding: utf-8

import json

import os
import requests
import copy
import time

template = {
	"model": "Qwen2.5-3B-Instruct",
	"prompt": "",
	"max_tokens": 600,
	"temperature": 0,
	"top_p": 0.9,
	"seed": 1112,
	"response_format": {
		"type": "json_object"
	},
	"stream": False
};

# "return json only,check syslogs,if some error happened alerm it by format '[{\"c\":\"lark\",\"p\":[msg,level]}]' "

sysprompt = """- You are syslog maintenance coder
- I am json CPU have 2 instructions:
1.PASS: all of syslog need not alarm.
{"c":"PASS"}
2. ALARM: some of syslog need alarm.
{ "c" : "LARK" , "syslog" : trigger in requests , "p" : [ describe as string , level as int ] }

You will code me with a json list and some instruction in it.
"""
# 3. if no alarm return {"c":"PASS"}
# 4. if have alarm return { "c" : "LARK" , "syslog" : trigger , "p" : [ string describe in chinese , int level ] } ,then level is enum LOW / HIGH / CRITICAL / EMERGENCY

sysprompt = """you are syslog checker
reply json only
- I have 2 instructions:
1. PASS: all of syslog need not alarm.
{"c":"PASS"}
2. ALARM: some of syslog need alarm.
{ "c" : "LARK" , "syslog" : trigger in requests , "p" : [ describe as string , level as int ] }
"""

def cprint(colorname,*text):
	colors = {
		'black': '\033[30m',
		'red': '\033[31m',
		'green': '\033[32m',
		'yellow': '\033[33m',
		'blue': '\033[34m',
		'magenta': '\033[35m',
		'cyan': '\033[36m',
		'white': '\033[37m'
	}
	if colorname in colors:
		print(colors[colorname],*text,"\033[0m")
	else:
 		print('\033[37}m',*text,"\033[0m")

# 颜色代码表（常见颜色）


def checkSyslog(syslog):
	target_url = "http://127.0.0.1:8080/v1/completions"
	question_for_search = copy.deepcopy(template);

	messageobj = [
		{
			"role":"system",
			"content":sysprompt
		},
		{
			"id":"1",
			"role":"user",
			"content":syslog
		}
	]

	question_for_search["prompt"] = json.dumps(messageobj);

	body = json.dumps(question_for_search,indent=4);
	print(body);
	r = requests.post(target_url,data=body);
	ret = json.loads(r.text)["content"];
	cprint("cyan",ret);

	pos = ret.find("```json");
	if pos != -1:
		ret1 = ret[pos+7:]
		pos = ret1.find("```");
		if pos != -1:
			ret = ret1[:pos];
	try:
		return json.loads(ret)
	except Exception as e:
		print(e);
		pass;
	return None;


txtsblocks = [];
with open("/tmp/syslog","rt") as f:
	for l in f:
		txtsblocks.append(l);
		
		if len(txtsblocks) > 10:
			print("这是一个块");
			tstart = time.time();
			alarmobj = checkSyslog("".join(txtsblocks));
			tend = time.time();
			cprint("yellow","耗时:",tend-tstart,"\n",json.dumps(alarmobj,indent=4));
			txtsblocks = [];