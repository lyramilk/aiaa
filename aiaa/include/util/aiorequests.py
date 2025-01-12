#-*- coding:utf-8 -*-

import aiohttp
from typing import Optional, Dict, Any

async def post(url: str, params: Optional[Dict[str, Any]] = None, data: str = None,json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,timeout: Optional[tuple[int, int]] = (1,5)) -> str:
	"""
	异步发送 POST 请求，返回文本格式（UTF-8 编码）
	:param url: 目标 URL
	:param json: 请求体（JSON 数据）
	:param headers: 请求头
	:return: 响应文本
	"""
	tmobj = aiohttp.ClientTimeout(total=timeout[1],connect=timeout[0]);
	async with aiohttp.ClientSession() as session:
		async with session.post(url, params=params, data=data, json=json, headers=headers, timeout=tmobj) as response:
			return await response.text(encoding="utf-8")

async def get(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,timeout: Optional[tuple[int, int]] = (1,5)) -> str:
	"""
	异步发送 GET 请求，返回文本格式（UTF-8 编码）
	:param url: 目标 URL
	:param params: 查询参数
	:param headers: 请求头
	:return: 响应文本
	"""
	tmobj = aiohttp.ClientTimeout(total=timeout[1],connect=timeout[0]);
	async with aiohttp.ClientSession() as session:
		async with session.get(url, params=params, headers=headers, timeout=tmobj) as response:
			return await response.text(encoding="utf-8")

async def put(url: str, data: str = None, json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,timeout: tuple[int, int] = (1,5)) -> str:
	"""
	异步发送 PUT 请求，返回文本格式（UTF-8 编码）
	:param url: 目标 URL
	:param json: 请求体（JSON 数据）
	:param headers: 请求头
	:return: 响应文本
	"""
	tmobj = aiohttp.ClientTimeout(total=timeout[1],connect=timeout[0]);
	async with aiohttp.ClientSession() as session:
		async with session.put(url, data=data, json=json, headers=headers, timeout=tmobj) as response:
			return await response.text(encoding="utf-8")

async def delete(url: str, headers: Optional[Dict[str, str]] = None,timeout: tuple[int, int] = (1,5)) -> str:
	"""
	异步发送 DELETE 请求，返回文本格式（UTF-8 编码）
	:param url: 目标 URL
	:param headers: 请求头
	:return: 响应文本
	"""
	tmobj = aiohttp.ClientTimeout(total=timeout[1],connect=timeout[0]);
	async with aiohttp.ClientSession() as session:
		async with session.delete(url, headers=headers, timeout=tmobj) as response:
			return await response.text(encoding="utf-8")