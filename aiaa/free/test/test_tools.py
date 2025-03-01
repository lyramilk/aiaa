from openai import OpenAI
import os

def send_messages(messages):
	response = client.chat.completions.create(
		model="deepseek-chat",
		messages=messages,
		tools=tools
	)
	return response.choices[0].message

client = OpenAI(
	api_key=os.environ.get("DEEPSEEK_APIKEY"),
	#base_url="https://api.deepseek.com",
	base_url="http://127.0.0.1:8888",
)

tools = [
	{
		"type": "function",
		"function": {
			"name": "get_weather",
			"description": "Get weather of an location, the user shoud supply a location first",
			"parameters": {
				"type": "object",
				"properties": {
					"location": {
						"type": "string",
						"description": "The city and state, e.g. San Francisco, CA",
					}
				},
				"required": ["location"]
			},
		}
	},
]

messages = [{"role": "user", "content": "How's the weather in Hangzhou?"}]
message = send_messages(messages)
print(f"User>\t {messages[0]['content']}")

tool = message.tool_calls[0]
messages.append(message)

messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24℃"})
message = send_messages(messages)
print(f"Model>\t {message.content}")