# gmail_sender.py

# pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
# if failed...
# pip uninstall google-api-python-client google-auth-httplib2 google-auth-oauthlib -y
# again...
# pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib


import os
import base64
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from common import display

class GmailSender:
    """
    Gmailを使ってメールを送信するためのモジュール。
    OAuth 2.0を使用して安全に認証を行います。
    """
    
    def __init__(self, token_path='token.json', client_secret_path='client_secret.json'):
        """
        初期化メソッド。認証に必要なファイルのパスを設定します。
        
        Args:
            token_path (str): 認証トークンを保存するファイルのパス。
            client_secret_path (str): Google Cloudからダウンロードした認証情報のパス。
        """
        self.token_path = token_path
        self.client_secret_path = client_secret_path
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.send']
        self.service = self._authenticate()

    def _authenticate(self):
        """
        OAuth 2.0フローを実行し、認証サービスを構築します。
        """
        creds = None
        # token.jsonが存在すれば、そこから認証情報をロード
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)
        
        # 認証情報がない、または無効な場合、新しいトークンを取得
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.client_secret_path, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # 取得した認証情報をtoken.jsonに保存
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())

        try:
            # Gmail APIサービスを構築
            return build('gmail', 'v1', credentials=creds)
        except HttpError as error:
            display.error('gmail_sender/_authenticate', f'{error}')
            return None

    def create_message(self, sender, to, subject, body):
        """
        メールメッセージを作成します。
        
        Args:
            sender (str): 送信元メールアドレス。
            to (str): 送信先メールアドレス。
            subject (str): メールの件名。
            body (str): メールの本文。
        
        Returns:
            dict: 送信可能な形式のメッセージ。
        """
        message = MIMEText(body)
        message['to'] = to
        message['from'] = sender
        message['subject'] = subject
        return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

    def send_message(self, sender, to, subject, body):
        """
        指定された情報でメールを送信します。
        
        Args:
            sender (str): 送信元メールアドレス。
            to (str): 送信先メールアドレス。
            subject (str): メールの件名。
            body (str): メールの本文。
        
        Returns:
            dict: APIレスポンス。
        """
        if not self.service:
            print("Authentification service is invalid. The mail cannot be sent.")
            return None
        
        message = self.create_message(sender, to, subject, body)
        
        try:
            send_message = self.service.users().messages().send(
                userId='me', body=message).execute()
            print(f"Sent mail. Message Id: {send_message['id']}")
            return send_message
        except HttpError as error:
            display.error('gmail_sender/send_message', f'{error}')
            return None
        