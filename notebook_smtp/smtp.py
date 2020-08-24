import smtplib
from email.header import Header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender = 'yourname@163.com'
receiver = 'yournam@xxx.com'
smtpserver = 'smtp.163.com'
username = 'yourname'
password = 'xxxxxx'  # 是授权密码而不是邮箱登陆密码，可参考https://github.com/hlmiao/Sagemakerlab/blob/master/notebook_smtp/authorization_code.png
mail_title = 'This a test email from Sagemaker'

# 创建一个带附件的实例
message = MIMEMultipart()
message['From'] = sender
message['To'] = receiver
message['Subject'] = Header(mail_title, 'utf-8') # 内容, 格式, 编码

# 邮件正文内容
message.attach(MIMEText('邮件正文', 'plain', 'utf-8'))

# 构造附件1（附件为log格式的文本，文件需要和该脚本放在相同目录）
att1 = MIMEText(open('jupyter.log', 'rb').read(), 'base64', 'utf-8')
att1["Content-Type"] = 'application/octet-stream'
att1["Content-Disposition"] = 'attachment; filename="jupyter.log"'
message.attach(att1)

# 构造附件2（附件为JPG格式的图片，文件需要和该脚本放在相同目录）
att2 = MIMEText(open('123.jpg', 'rb').read(), 'base64', 'utf-8')
att2["Content-Type"] = 'application/octet-stream'
att2["Content-Disposition"] = 'attachment; filename="123.jpg"'
message.attach(att2)

smtpObj = smtplib.SMTP_SSL()  # 启用SSL发信, 端口一般是465
smtpObj.connect(smtpserver)
smtpObj.login(username, password)
smtpObj.sendmail(sender, receiver, message.as_string()) # 发送
print("邮件发送成功！！！")
smtpObj.quit()
