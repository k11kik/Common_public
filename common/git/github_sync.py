import requests
import base64
import os
from datetime import datetime
import sys
import time
from common import display


# -----------------------------------------------------------
# download
# -----------------------------------------------------------
def collect_github_files(owner, repo, branch, dir_path, github_token):
    """
    指定ディレクトリ以下の全ファイルパスを再帰的に収集
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{dir_path}?ref={branch}"
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    r = requests.get(api_url, headers=headers)
    if r.status_code != 200:
        raise Exception(f"GitHub API error: {r.status_code} {r.text}")
    items = r.json()
    # __pycache__を除外
    items = [item for item in items if not (item['type'] == 'dir' and item['name'] == '__pycache__')]
    file_list = []
    for item in items:
        if item['type'] == 'file':
            file_list.append(item['path'])
        elif item['type'] == 'dir':
            file_list.extend(collect_github_files(owner, repo, branch, item['path'], github_token))
    return file_list


def _download_github_file_api(
    owner, repo, branch, file_path, local_base_dir, github_token,
    info: bool = True,
    subinfo: bool = True,
    max_retries: int = 3
):
    """単一ファイルをGitHub API経由でダウンロードし、リトライ機能を組み込む (最大3回)"""
    for attempt in range(max_retries):
        try:
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}"
            headers = {}
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            r = requests.get(api_url, headers=headers)
            
            if r.status_code == 200:
                content = r.json()
                file_content = base64.b64decode(content["content"])
                local_path = os.path.join(local_base_dir, file_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(file_content)
                if subinfo:
                    print(f"Downloaded: {file_path} -> {local_path}")
                return local_path
            
            # エラー処理
            if attempt < max_retries - 1:
                print(f"Error downloading {file_path}: Status {r.status_code}. Retrying in 1s... (Attempt {attempt + 2}/{max_retries})")
                time.sleep(1)
                continue
            else:
                raise Exception(f"GitHub API error after {max_retries} attempts for {file_path}: {r.status_code} {r.text}")

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Connection error downloading {file_path}: {e}. Retrying in 1s... (Attempt {attempt + 2}/{max_retries})")
                time.sleep(1)
                continue
            else:
                raise Exception(f"Connection error downloading {file_path} after {max_retries} attempts: {e}")
    return local_path


def download(
    owner: str,
    repo: str,
    remote_path: str,
    github_token: str,
    branch: str = 'main',
    is_dir: bool = None,
    local_base_dir: str = '.',
    confirm: bool = True,
    info: bool = True,
    subinfo: bool = False,
    extensions: list[str] | None = None
):
    """
    指定したGitHubのファイルまたはディレクトリをダウンロードする汎用関数。

    Parameters
    ----------
    owner : str
        GitHubリポジトリのオーナー名。
    repo : str
        GitHubリポジトリ名。
    branch : str
        ブランチ名。
    remote_path : str
        GitHub上のファイルまたはディレクトリパス。
    is_dir : bool
        ダウンロード対象がディレクトリである場合はTrue。ファイルの場合はFalse。
    local_base_dir : str
        ローカルの保存先ベースディレクトリ。
    github_token : str
        GitHub Personal Access Token。
    confirm : bool
        ダウンロード実行前に確認を求めるかどうか。
    info : bool
        進捗情報を表示するかどうか。
    subinfo : bool
        詳細なサブ情報を表示するかどうか。
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{remote_path}?ref={branch}"
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    r = requests.get(api_url, headers=headers)
    if r.status_code != 200:
        raise Exception(f"GitHub API error while detecting path type: {r.status_code} {r.text}")
    
    res_data = r.json()

    # is_dir が未指定(None)の場合、レスポンスデータがリスト構造（複数ファイル情報）ならディレクトリと判定
    if is_dir is None:
        is_dir = isinstance(res_data, list)

    if is_dir:
        # ディレクトリダウンロードの場合
        file_list = collect_github_files(owner, repo, branch, remote_path, github_token)
    else:
        # ファイルダウンロードの場合
        file_list = [remote_path]

    # filtering by the extensions
    if extensions is None:
        extensions = ['.py']
    if extensions:
        lower_extensions = [ext.lower() for ext in extensions]
        original_count = len(file_list)
        filtered_list = []
        for file_path in file_list:
            if any(file_path.lower().endswith(ext) for ext in lower_extensions):
                filtered_list.append(file_path)
        file_list = filtered_list
        if subinfo and original_count > len(file_list):
            print(f"Filtered files: {original_count} -> {len(file_list)} (Extensions: {', '.join(extensions)})")

    # 確認ステップ
    print("\n[Download plan]")
    print('-' * 50)
    print(f'Owner: {owner}')
    print(f'Repository: {repo}')
    print(f'Branch: {branch}')
    print(f'Remote path: {remote_path}')
    print('-' * 50)

    for i, path in enumerate(file_list):
        local_path = os.path.join(local_base_dir, path)
        print(f"{i+1}. {path} -> {local_path}")
    if confirm and info:
        ans = input(f"Total files: {len(file_list)}. Proceed with download? [Y/n]: ").strip().lower()
        if ans not in ['', 'y']:
            print("Download cancelled.")
            return

    # ダウンロード実行
    if not file_list:
        if info:
            print("No files to download.")
        return []

    downloaded_files = []
    success_count = 0 # 成功数をカウントする変数を追加
    total_files = len(file_list)
    start_time = datetime.now()
    for idx, file_path in enumerate(file_list):
        if info:
            display.progress_bar(idx, len(file_list), start_time)
        try:
            local_path = _download_github_file_api(owner, repo, branch, file_path, local_base_dir, github_token, subinfo=subinfo)
            downloaded_files.append(local_path)
            success_count += 1 # success
        except Exception as e:
            print(f"\n[Error] Failed to download '{file_path}': {e}", file=sys.stderr)
    
    # if info:
    #     display.progress_bar(len(file_list), len(file_list), start_time, end=True)
    #     print("Download complete.")

    if info:
        display.current_time_comment(comment=f'Download complete: {success_count}/{total_files}')

    return downloaded_files


# ---------------------------------------------------------
# upload
# ---------------------------------------------------------


def upload_github_file(
    github_repo: str,         # 例: "k11kik/Messenger"
    branch: str,              # 例: "main"
    remote_path: str,         # 例: "messenger_analysis/newfile.py"
    local_file_path: str,     # 例: "./newfile.py"
    commit_message: str,      # 例: "Add newfile.py"
    github_token: str,
    info: bool = True,
    subinfo: bool = True,
    max_retries: int = 3
):
    """
    指定したローカルファイルをGitHubリポジトリの指定パスにアップロード（新規/上書き）する。
    失敗した場合は最大3回リトライを試行する。
    """
    for attempt in range(max_retries):
        try:
            # 1. ファイル内容をbase64エンコード
            with open(local_file_path, "rb") as f:
                content = base64.b64encode(f.read()).decode("utf-8")

            # 2. 既存ファイルのSHAを取得（上書きの場合のみ必要）
            api_url = f"https://api.github.com/repos/{github_repo}/contents/{remote_path}"
            headers = {"Authorization": f"token {github_token}"}
            params = {"ref": branch}
            
            sha = None
            sha_r = requests.get(api_url, headers=headers, params=params)
            if sha_r.status_code == 200:
                sha = sha_r.json()["sha"]
            
            # 3. アップロード（PUTリクエスト）
            data = {
                "message": commit_message,
                "content": content,
                "branch": branch
            }
            if sha:
                data["sha"] = sha

            r = requests.put(api_url, headers=headers, json=data)
            
            if r.status_code in (200, 201):
                if subinfo:
                    print(f"Uploaded: {local_file_path} -> {remote_path}")
                return # 成功
            
            # エラー処理
            if attempt < max_retries - 1:
                print(f"Error uploading {local_file_path}: Status {r.status_code}. Retrying in 1s... (Attempt {attempt + 2}/{max_retries})")
                time.sleep(1)
                continue
            else:
                # 最終試行で失敗
                raise Exception(f"GitHub API error after {max_retries} attempts for {local_file_path}: {r.status_code} {r.text}")

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Connection error uploading {local_file_path}: {e}. Retrying in 1s... (Attempt {attempt + 2}/{max_retries})")
                time.sleep(1)
                continue
            else:
                # 最終試行で接続エラー
                raise Exception(f"Connection error uploading {local_file_path} after {max_retries} attempts: {e}")


def upload(
    owner: str,
    repo: str,
    local_path: str,
    github_token: str,
    branch: str = 'main',
    extensions: list[str] | None = None,
    remote_path: str = None,
    is_dir: bool = None,
    commit_message: str = 'Update via script',
    confirm: bool = True,
    subinfo: bool = False,
):
    """
    指定したローカルのファイルまたはディレクトリをGitHubにアップロードする汎用関数。

    Parameters
    ----------
    owner : str
        GitHubリポジトリのオーナー名。
    repo : str
        GitHubリポジトリ名。
    branch : str
        ブランチ名。
    local_path : str
        アップロードするローカルのファイルまたはディレクトリパス。
    remote_path : str
        GitHub上のアップロード先パス。
    is_dir : bool
        アップロード対象がディレクトリである場合はTrue。ファイルの場合はFalse。
    commit_message : str
        GitHubへのコミットメッセージ。
    github_token : str
        GitHub Personal Access Token。
    confirm : bool
        アップロード実行前に確認を求めるかどうか。
    info : bool
        進捗情報を表示するかどうか。
    subinfo : bool
        詳細なサブ情報を表示するかどうか。
    """
    # if github_token is None:
    #     raise ValueError("A GitHub token is required for upload operations.")
    
    if extensions is None:
        extensions = ['.py']

    file_pairs = []
    lower_extensions = [ext.lower() for ext in extensions] if extensions else None

    # local path
    abs_local_path = os.path.abspath(local_path)
    base_name = os.path.basename(abs_local_path)

    if is_dir is None:
        is_dir = os.path.isdir(local_path)

    if is_dir:
        if not remote_path:
            target_remote_dir = base_name
        else:
            target_remote_dir = os.path.join(remote_path, base_name).replace("\\", "/")

        for root, dirs, files in os.walk(local_path):
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')
            for file in files:
                # 拡張子フィルタリング (extensionsがNoneまたは空の場合は全ファイル対象)
                is_match = True
                if lower_extensions:
                    is_match = any(file.lower().endswith(ext) for ext in lower_extensions)
                
                if is_match:
                    local_file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_file_path, local_path)
                    # remote_pathがディレクトリ名を含む場合があるため、os.path.joinで結合
                    remote_file_path = os.path.join(target_remote_dir, rel_path).replace("\\", "/")
                    file_pairs.append((local_file_path, remote_file_path))

    else:
        # 単一ファイルのアップロード処理
        if not remote_path:
            # remote_pathが未指定（Noneまたは空）の場合は、最上位ルートに直接配置
            remote_file_path = base_name
        else:
            # 指定されたremote_pathが拡張子を含んでいるか、フォルダ名のみかを自動アライン
            is_remote_file = remote_path.endswith('.py') or any(remote_path.lower().endswith(ext) for ext in (lower_extensions or []))
            if is_remote_file:
                remote_file_path = remote_path.replace("\\", "/")
            else:
                remote_file_path = os.path.join(remote_path, base_name).replace("\\", "/")
                
        file_pairs.append((abs_local_path, remote_file_path))

        # # 単一ファイル
        # remote_file_path = os.path.join(remote_path, os.path.basename(local_path)).replace("\\", "/")
        # file_pairs.append((local_path, remote_file_path))

    # 確認ステップ
    print("\n[Upload plan]")
    print('-' * 50)
    print(f'Owner: {owner}')
    print(f'Repository: {repo}')
    print(f'Branch: {branch}')
    print(f'Local path: {local_path}')
    print('-' * 50)
    for i, (local_file, remote_file) in enumerate(file_pairs):
        print(f"{i+1}. {local_file} -> {remote_file}")
    if confirm:
        ans = input(f"Total files: {len(file_pairs)}. Proceed with upload? [Y/n]: ").strip().lower()
        if ans not in ['', 'y']:
            print("Upload cancelled.")
            return

    # アップロード実行
    if not file_pairs:
        display.info('No files to upload.')
        return

    github_repo = f"{owner}/{repo}"
    total = len(file_pairs)
    success_count = 0
    start_time = datetime.now()
    for idx, (local_file_path, remote_file_path) in enumerate(file_pairs):
        display.progress_bar(idx, total, start_time)
        try:
            upload_github_file(
                github_repo=github_repo,
                branch=branch,
                remote_path=remote_file_path,
                local_file_path=local_file_path,
                commit_message=commit_message,
                github_token=github_token,
                subinfo=subinfo
            )
            success_count += 1
        except Exception as e:
            print(f"\nError uploading {local_file_path}: {e}", file=sys.stderr)
            # エラー発生時も続行
            
    # if info:
    #     display.progress_bar(total, total, start_time, end=True)
    #     print("Upload complete.")

    display.info(f'Upload complete: {success_count}/{total}')

    return_message = (
        f'{owner} (Branch: {branch}): {local_path} -> {repo}\n'
        f'  Uploaded: {success_count}/{total}'
    )
    return return_message

