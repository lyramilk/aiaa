# -*- coding: utf-8 -*-


import os
import json
import requests
from colorama import Fore, Back, Style, init

appkey = os.environ.get('DEEPSEEK_APIKEY');


def translate_code(code_type,code_text):
	return code_text;
	'''
	代码中的注释翻译的时候有的时候会导致代码截断，这个就不翻译了。
	tmp = {
		"model": "deepseek-chat",
		"messages": [
			{
				"role": "system",
				"content": "翻译下面" + code_type + "代码中的注释部分，不要翻译字符串，其它部分保持原样。不要添加额外的markdown标记"
			},
			{
				"role": "user",
				"content": "\n".join(code_text),
			}
		],
		"stream": False,
		"max_tokens": 8192,
	}
	r = requests.post("https://api.deepseek.com/v1/chat/completions", data=json.dumps(tmp),headers={"Content-Type": "application/json","Authorization": "Bearer " + appkey});
	ro = r.json();
	print("把",code_type,"代码\n",Fore.RED,"\n".join(code_text),Style.RESET_ALL,"\n");
	print("翻译为\n",Fore.GREEN,ro["choices"][0]["message"]["content"],Style.RESET_ALL,"\n");
	return [x + "\n" for x in ro["choices"][0]["message"]["content"].split("\n")]
	'''

def translate_markdown(markdown_text):
	tmp = {
		"model": "deepseek-chat",
		"messages": [
			{
				"role": "system",
				"content": "翻译下面文字部分，保留原始的html、markdown、katex代码格式。不要添加额外的markdown标记。只翻译给定的文字不要联想没有提及的内容。"
			},
			{
				"role": "user",
				"content": "\n".join(markdown_text),
			}
		],
		"stream": False,
		"max_tokens": 8192,
	}
	#print(json.dumps(tmp));
	r = requests.post("https://api.deepseek.com/v1/chat/completions", data=json.dumps(tmp),headers={"Content-Type": "application/json","Authorization": "Bearer " + appkey});
	ro = r.json();
	print("把markdown\n",Fore.RED,"\n".join(markdown_text),Style.RESET_ALL,"\n");
	print("翻译为\n",Fore.GREEN,ro["choices"][0]["message"]["content"],Style.RESET_ALL,"\n");
	return [x + "\n" for x in ro["choices"][0]["message"]["content"].split("\n")]

def translate_ipynb(src,dest):
	print(src,dest);

	with open(src,"r") as fr:
		with open(dest,"w") as fw:
			ipynb_obj = json.loads(fr.read());
			
			code_type = ipynb_obj["metadata"]["kernelspec"]["language"]
			
			zh_CN_cells = [];
			for cell in ipynb_obj["cells"]:
				cell_type = cell["cell_type"]
				if cell_type == "markdown":
					newcell = cell.copy();
					newcell["source"] = translate_markdown(cell["source"]);
					zh_CN_cells.append(newcell);
				elif cell_type == "code":
					newcell = cell.copy();
					newcell["source"] = translate_code(code_type,cell["source"]);
					zh_CN_cells.append(newcell);
				else:
					raise Exception("未识别的celltype:",cell_type);

			ipynb_obj["cells"] = zh_CN_cells;
			
			fw.write(json.dumps(ipynb_obj,indent=2));
			



import sys
ipynb_name = sys.argv[1];

translate_ipynb(ipynb_name,ipynb_name.replace(".ipynb",".zh_CN.ipynb"));

