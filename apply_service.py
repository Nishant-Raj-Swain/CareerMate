import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

def send_application(draft: dict):
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")

    if not username or not password:
        raise ValueError("SMTP credentials not configured on server.")

    msg = MIMEMultipart()
    msg["From"] = username
    msg["To"] = draft["recipient"]
    msg["Subject"] = draft["subject"]
    msg.attach(MIMEText(draft["body"], "plain"))

    attachment = MIMEApplication(draft["resume"].encode("utf-8"), _subtype="txt")
    attachment.add_header("Content-Disposition", "attachment", filename="resume.txt")
    msg.attach(attachment)

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(msg)