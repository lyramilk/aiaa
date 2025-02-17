from typing import List, Dict, Optional, Generator, Union, Callable, Any
import json
import requests
from sentence_transformers import SentenceTransformer
import faiss
import os
from modelscope import snapshot_download
import transformers

class AIChannel:
    """AI对话管理器，支持长对话历史和工具调用"""
    
    def __init__(self, host:str = "https://api.deepseek.com", max_context_length: int = 131072):
        """初始化对话管理器"""
        self.host = host

        self.max_context_length = max_context_length
        self.system_prompt = "You are a helpful assistant."
        
        # 对话存储
        self.recent_history: List[Dict[str, str]] = []  # 最近几轮对话
        self.context: List[Dict[str, str]] = []         # 当前上下文
        
        # 向量存储
        model_dir = snapshot_download('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_dir)
        self.index = faiss.IndexFlatIP(384)  # 向量维度384
        self.message_store: List[Dict[str, str]] = []   # 所有历史消息
        
        # 工具配置
        self.tools: Dict[str, Dict] = {}
        self.tool_implementations: Dict[str, Callable] = {}
        
        # 延迟初始化
        self._tokenizer = None
        
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if self.host == "https://api.deepseek.com":
                self._tokenizer = DeepSeekTokenizer()
            else:   
                self._tokenizer = DefaultTokenizer()
        return self._tokenizer

    def add_message(self, role: str, content: str) -> None:
        """添加新消息到对话历史"""
        message = {'role': role, 'content': content}
        
        # 更新最近对话历史(保留最近3轮)
        self.recent_history.append(message)
        if len(self.recent_history) > 6:  # 3轮 * 2条消息/轮
            self.recent_history = self.recent_history[-6:]
            
        # 更新向量存储
        embedding = self.embedding_model.encode(content).reshape(1, -1)
        self.index.add(embedding)
        self.message_store.append(message)

    def get_context(self, query: str) -> List[Dict[str, str]]:
        """获取当前查询的相关上下文"""
        context = []
        remaining_tokens = self.max_context_length
        
        # 添加最近对话
        for msg in reversed(self.recent_history):
            msg_tokens = self.tokenizer.count_message_tokens(msg)
            if remaining_tokens >= msg_tokens:
                context.insert(0, msg)
                remaining_tokens -= msg_tokens
        
        # 添加相关历史消息
        if remaining_tokens > 0:
            relevant = self._get_relevant_messages(query, top_k=5, min_score=0.6)
            for msg in relevant:
                if msg not in context:
                    msg_tokens = self.tokenizer.count_message_tokens(msg)
                    if remaining_tokens >= msg_tokens:
                        context.insert(0, msg)
                        remaining_tokens -= msg_tokens
        
        self.context = context
        return context

    def _get_relevant_messages(self, query: str, top_k: int, min_score: float) -> List[Dict[str, str]]:
        """检索相关历史消息"""
        if not self.message_store:
            return []
            
        query_embedding = self.embedding_model.encode(query).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)
        
        return [
            self.message_store[idx] 
            for score, idx in zip(scores[0], indices[0])
            if idx != -1 and score >= min_score
        ]

    def add_tool(self, name: str, func: Callable, description: str, parameters: Dict[str, Dict[str, Any]]) -> None:
        """注册工具"""
        self.tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": list(parameters.keys())
                }
            }
        }
        self.tool_implementations[name] = func

    def call(self, message: str, json_mode: bool = False, **kwargs) -> Union[str, Dict]:
        """发送消息并获取回复"""
        self.add_message('user', message)
        context = self.get_context(message)
        
        response = self._send_request(context, json_mode, **kwargs)
        
        if 'tool_calls' in response['choices'][0]['message']:
            return self._handle_tool_calls(response, json_mode, **kwargs)
            
        content = response['choices'][0]['message']['content']
        self.add_message('assistant', content)
        
        return json.loads(content) if json_mode else content

    def _send_request(self, context: List[Dict[str, str]], json_mode: bool, **kwargs) -> Dict:
        """发送API请求"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_APIKEY')}"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": self.system_prompt}] + context,
            "tools": list(self.tools.values()) if self.tools else None,
            "response_format": {"type": "json_object"} if json_mode else None,
            "stream": False,
            "max_tokens": 8192,
            **kwargs
        }
        
        response = requests.post(
            self.host + "/v1/chat/completions",
            headers=headers,
            json=data
        )
        return response.json()

    def _handle_tool_calls(self, response: Dict, json_mode: bool, **kwargs) -> Union[str, Dict]:
        """处理工具调用"""
        tool_calls = response['choices'][0]['message']['tool_calls']
        results = []
        
        for call in tool_calls:
            func_name = call['function']['name']
            func_args = json.loads(call['function']['arguments'])
            
            try:
                result = self.tool_implementations[func_name](**func_args)
                results.append({
                    'tool_call_id': call['id'],
                    'name': func_name,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'tool_call_id': call['id'],
                    'name': func_name,
                    'error': str(e)
                })
        
        self.add_message('assistant', json.dumps(response['choices'][0]['message']))
        for result in results:
            self.add_message('tool', json.dumps(result))
            
        return self.call(
            "Please process the tool results and continue our conversation.",
            json_mode,
            **kwargs
        )

    def clear(self) -> None:
        """清除所有历史记录"""
        self.recent_history.clear()
        self.context.clear()
        self.message_store.clear()
        self.index.reset()

    def stream_call(self, message: str, json_mode: bool = False, **kwargs) -> Generator[str, None, None]:
        """流式发送消息并获取回复"""
        self.add_message('user', message)
        context = self.get_context(message)
        
        # 准备请求数据
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_APIKEY')}"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": self.system_prompt}] + context,
            "tools": list(self.tools.values()) if self.tools else None,
            "response_format": {"type": "json_object"} if json_mode else None,
            "stream": True,  # 启用流式响应
            **kwargs
        }
        
        # 发送流式请求
        response = requests.post(
            self.host + "/v1/chat/completions",
            headers=headers,
            json=data,
            stream=True
        )
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]  # 移除 "data: " 前缀
                if line == '[DONE]':
                    break
                    
                try:
                    chunk = json.loads(line)
                    if chunk.get('choices'):
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            content = delta['content']
                            full_response += content
                            yield content
                except json.JSONDecodeError:
                    continue
        
        # 保存完整响应到历史记录
        self.add_message('assistant', full_response)

class DeepSeekTokenizer:
	"""DeepSeek模型的tokenizer封装类"""

	def __init__(self):
		"""初始化tokenizer"""
		model_dir = snapshot_download('lyramilk/deepseek_v3_tokenizer')
		self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, revision='v1.0.0')

	def count_tokens(self, text: str) -> int:
		"""
		计算文本的token数量
		:param text: 输入文本
		:return: token数量
		"""
		return len(self.tokenizer.encode(text))

	def count_message_tokens(self, message: dict) -> int:
		"""
		计算单条消息的token数量
		:param message: 消息字典，包含role和content
		:return: token数量
		"""
		role = message['role']
		content = message['content']
		# 计算格式化后的完整消息长度
		formatted_message = f"{role}: {content}"
		return self.count_tokens(formatted_message) 
    
class DefaultTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text)

    def count_message_tokens(self, message: dict) -> int:
        return len(message['content'])