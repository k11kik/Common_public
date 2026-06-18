import os
import sys
import json
import platform
from common import display


def load_github_token(
        token_path: str = None
    ) -> str:
    """
    Parameters
    ----------
    toke_path : str, default "/Users/username/.github_token.json"
        読み込むファイル名。
        拡張子が ".json" の場合はJSONパーサーを、それ以外はテキストとして自動で適用します。

    Returns
    -------
    str
        取得されたGitHubトークン文字列。取得できなかった場合は空文字列を返します。
    """
    if token_path is None:
        current_os = platform.system()
    
        if current_os == "Windows":
            # Windowsのユーザーホーム (例: C:\Users\username)
            home_dir = os.environ.get("USERPROFILE") or os.path.expanduser("~")
        else:
            # Mac / Linux のユーザーホーム (例: /Users/username)
            home_dir = os.environ.get("HOME") or os.path.expanduser("~")
        
        token_path = os.path.join(home_dir, '.github_token.json')
    
    if not os.path.exists(token_path):
        display.warning(f'Not found: {token_path}')
        return ""

    try:
        # 拡張子の判定
        _, ext = os.path.splitext(token_path)
        
        # JSON形式ファイルの場合の処理
        if ext.lower() == '.json':
            with open(token_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                token = config.get('github_token') or config.get('token')
                if token:
                    return str(token).strip()
                
        # テキストファイル形式の場合の処理 (.txt、またはドットファイルなど)
        else:
            with open(token_path, 'r', encoding='utf-8') as f:
                token = f.readline().strip()
                if token:
                    return token

        print(f"[Warning] Token format is invalid or empty in: {token_path}", file=sys.stderr)
        return ""

    except Exception as e:
        print(f"[Error] Failed to read token from {token_path}: {e}", file=sys.stderr)
        return ""

