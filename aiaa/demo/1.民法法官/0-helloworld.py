import aiohttp.web

async def handle_index(request):
	return aiohttp.web.Response(text="Hello AI world!", status=200)

# 创建应用程序对象
app = aiohttp.web.Application()
# 添加路由
app.router.add_get('/', handle_index)
aiohttp.web.run_app(app,host='0.0.0.0', port=8888)
