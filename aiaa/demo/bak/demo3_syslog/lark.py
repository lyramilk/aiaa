import requests
import json
import time


class LarkError(Exception):
	def __init__(self,str):
		super(Exception, self).__init__(str);


class document_reader:
	def __init__(self):
		self.appid = "cli_a30402a007b89009";
		self.appsec = "lhiE8GlWFFndNUvLdkmgO0CMFt12SD8N";
		self._cache = {};

	def get_tenant_access_token(self):
		cachekey = "tenant_access_token:" + self.appid;

		tenant_access_token = self.cache_get(cachekey);
		if tenant_access_token is None:
			url = "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal"
			body = {};
			body["app_id"] = self.appid;
			body["app_secret"] = self.appsec;

			ret = requests.post(url,data=json.dumps(body));

			obj = json.loads(ret.text);
			if obj["code"] == 0:
				self.cache_put(cachekey,obj["tenant_access_token"],obj["expire"]);
				return obj["tenant_access_token"];
			raise LarkError(ro["msg"]);

		return tenant_access_token;


	def get_app_token_by_wiki_token(self,wiki_token):
		hdr = {'Authorization':'Bearer %s'%self.get_tenant_access_token()};

		url = f'https://open.larksuite.com/open-apis/wiki/v2/spaces/get_node?token={wiki_token}'
		ret = requests.get(url,headers=hdr);
		ro = json.loads(ret.text);
		if ro["code"] == 0:
			return ro['data']['node']['obj_token']
		raise LarkError(ro["msg"]);

	def get_excel_meta(self,spreadsheetToken):
		if spreadsheetToken.startswith("wiku"):
			spreadsheetToken = self.get_app_token_by_wiki_token(spreadsheetToken);
		hdr = {'Authorization':'Bearer %s'%self.get_tenant_access_token()};

		url = f'https://open.larksuite.com/open-apis/sheets/v2/spreadsheets/{spreadsheetToken}/metainfo';
		ret = requests.get(url,headers=hdr);
		ro = json.loads(ret.text);
		if ro["code"] == 0:
			return ro["data"];
		raise LarkError(ro["msg"]);
	
	def search_wiki(self,query):
		url = "https://open.larksuite.com/open-apis/wiki/v1/nodes/search";
		hdr = {'Authorization':'Bearer %s'%self.get_tenant_access_token()};

		search_request_object = {
			"query": query,
		}
		ret = requests.post(url,headers=hdr,data=json.dumps(search_request_object));
		print(ret.text);
		ro = json.loads(ret.text);
		print(ro);


	def get_excel_data(self,spreadsheetToken,sheetId,range = None):
		if spreadsheetToken.startswith("wiku"):
			spreadsheetToken = self.get_app_token_by_wiki_token(spreadsheetToken);
		hdr = {'Authorization':'Bearer %s'%self.get_tenant_access_token()};

		if range is not None and len(range) > 0:
			url = f"https://open.larksuite.com/open-apis/sheets/v2/spreadsheets/{spreadsheetToken}/values/{sheetId}!{range}?valueRenderOption=ToString";
		else:
			url = f"https://open.larksuite.com/open-apis/sheets/v2/spreadsheets/{spreadsheetToken}/values/{sheetId}?valueRenderOption=ToString";
		ret = requests.get(url,headers=hdr);
		ro = json.loads(ret.text);
		if ro["code"] == 0:
			return ro["data"]["valueRange"]["values"];
		raise LarkError(ro["msg"]);


	def get_bitable_data(self,app_token,table_id,filter = None,page_size=500):
		if app_token.startswith("wiku"):
			app_token = self.get_app_token_by_wiki_token(app_token);


		hdr = {'Authorization':'Bearer %s'%self.get_tenant_access_token()};
		url = f"https://open.larksuite.com/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records";

		'''
		expstrs = [];
		for k in filter:
			v = filter[v];
			expstr = f'CurrentValue.[{k}]="{v}"'
			expstrs.append(expstr);
		params["filter"] = "AND(" + ",".join(expstrs) + ")"
		'''

		params = {};
		params["page_size"] = page_size;
		params["filter"] = filter;
		ret = requests.get(url,params=params,headers=hdr);
		ro = json.loads(ret.text);
		if ro["code"] == 0:
			rdata = [];
			for item in ro["data"]["items"]:
				rdata.append(item["fields"]);
			return rdata;
		raise LarkError(ro["msg"]);


	def easy_get_table(self,spreadsheetToken,sheetindex = 0):
		if spreadsheetToken.startswith("wiku"):
			spreadsheetToken = self.get_app_token_by_wiki_token(spreadsheetToken);
		wbk = self.get_excel_meta(spreadsheetToken);
		if sheetindex < len(wbk["sheets"]):
			return self.get_excel_data(spreadsheetToken,wbk["sheets"][sheetindex]["sheetId"]);
		return None;

	def easy_get_table_as_pandas(self,spreadsheetToken,sheetindex = 0):
		if spreadsheetToken.startswith("wiku"):
			spreadsheetToken = self.get_app_token_by_wiki_token(spreadsheetToken);
		import pandas as pd
		dt = self.easy_get_table(spreadsheetToken,sheetindex);
		return pd.DataFrame(data = dt[1:],columns=dt[0]);
		

	def cache_put(self,key,value,expire):
		cachekey = "tenant_access_token:" + self.appid;
		self._cache[key] = {"v":value,"e":int(time.time()) + expire}

	def cache_get(self,key):
		if key not in self._cache:
			return None;
		
		o = self._cache[key];
		if int(time.time()) > o["e"]:
			del self._cache[key];
			return None;
		return o["v"];


'''
t = lark.document_reader();
dt = t.easy_get_table("shtxxxxxxxxxxxxxxxxxxxxxxxx");
print(dt);


t = document_reader();
dt = t.get_bitable_data("ZTYfb91pba2EhQsvzXTu2DBosib","tblVDYuz9fi3fyhE",filter='CurrentValue.[服务] ="nct-player-playlist-service"');
print(dt);




t = document_reader();
dt = t.get_bitable_data("wikusR0VPLQ2IT4GGk7FWuqh6Ec","tblKVuthFP8NMSkS");
print(dt);
'''


t = document_reader();
t.search_wiki("");