# -*- coding: utf-8 -*-
# coding: utf-8

import json
import os
import copy
import requests
import chatsession





session = chatsession.ChatData();
session.prompt("""
你是一个产品经理，请根据描述分解需求。

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
""");

r = session.get();

print(json.dumps(r,indent=4));