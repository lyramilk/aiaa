import ssl
import socket
import datetime
import OpenSSL
from urllib.parse import urlparse
import os

def get_ssl_certificate_info(url):
    """
    获取指定URL的SSL证书信息
    
    Args:
        url (str): 完整的HTTPS URL或域名
        
    Returns:
        dict: 包含证书信息的字典，包括过期时间戳、颁发者、主题等
    """
    # 解析URL获取主机名
    if not url.startswith('http'):
        url = 'https://' + url
    
    parsed_url = urlparse(url)
    hostname = parsed_url.netloc
    
    # 如果主机名包含端口号，则去除端口号
    if ':' in hostname:
        hostname = hostname.split(':')[0]
    
    # 默认HTTPS端口
    port = 443
    
    try:
        # 创建SSL上下文
        context = ssl.create_default_context()
        
        # 连接到服务器
        with socket.create_connection((hostname, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                # 获取证书
                cert_binary = ssock.getpeercert(binary_form=True)
                cert = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_ASN1, cert_binary)
                
                # 获取证书过期时间
                not_after = cert.get_notAfter().decode('ascii')
                expiration_date = datetime.datetime.strptime(not_after, '%Y%m%d%H%M%SZ')
                expiration_timestamp = int(expiration_date.timestamp())
                
                # 获取颁发者信息
                issuer = cert.get_issuer().get_components()
                issuer_dict = {k.decode('utf-8'): v.decode('utf-8') for k, v in issuer}
                
                # 获取主题信息
                subject = cert.get_subject().get_components()
                subject_dict = {k.decode('utf-8'): v.decode('utf-8') for k, v in subject}
                
                # 返回证书信息
                return {
                    'hostname': hostname,
                    'expiration_date': expiration_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'expiration_timestamp': expiration_timestamp,
                    'issuer': issuer_dict,
                    'subject': subject_dict,
                    'days_remaining': (expiration_date - datetime.datetime.now()).days
                }
    
    except Exception as e:
        return {
            'error': str(e),
            'hostname': hostname
        }

def is_certificate_expiring_soon(url, days_threshold=7):
    """
    检查证书是否即将过期
    
    Args:
        url (str): 完整的HTTPS URL或域名
        days_threshold (int): 过期警告阈值（天数）
        
    Returns:
        bool: 如果证书将在指定天数内过期，则返回True
    """
    cert_info = get_ssl_certificate_info(url)
    
    if 'error' in cert_info:
        print(f"获取证书信息时出错: {cert_info['error']}")
        return None
    
    return cert_info['days_remaining'] <= days_threshold

# 使用示例
if __name__ == "__main__":
    test_url = "www.lyramilk.com"
    cert_info = get_ssl_certificate_info(test_url)
    
    if 'error' in cert_info:
        print(f"获取证书信息时出错: {cert_info['error']}")
    else:
        print(f"域名: {cert_info['hostname']}")
        print(f"证书过期时间: {cert_info['expiration_date']}")
        print(f"过期时间戳: {cert_info['expiration_timestamp']}")
        print(f"剩余天数: {cert_info['days_remaining']}")
        
        if is_certificate_expiring_soon(test_url, 7):
            print("警告: 证书将在60天内过期!")
            os.system("sudo certbot renew --dry-run")
        else:
            print("证书有效期充足。")
