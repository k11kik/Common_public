import requests
import base64
import os
from datetime import datetime
import sys
import time
# import re
from .base import display


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


def download_github(
    owner: str,
    repo: str,
    branch: str,
    remote_path: str,
    github_token: str,
    is_dir: bool = False,
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
    if confirm and info:
        print("\n[Download plan]")
        if is_dir:
            print(f"Dir: {remote_path}")
        else:
            print(f"File: {remote_path}")

        for i, path in enumerate(file_list):
            local_path = os.path.join(local_base_dir, path)
            print(f"{i+1}. {path} -> {local_path}")
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


def upload_github(
    owner: str,
    repo: str,
    branch: str,
    local_path: str,
    remote_path: str,
    github_token: str,
    is_dir: bool = False,
    commit_message: str = 'Update via script',
    confirm: bool = True,
    info: bool = True,
    subinfo: bool = False,
    extensions: list[str] | None = None
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

    if is_dir:
        # ディレクトリ内の全ての.pyファイルを収集
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
                    remote_file_path = os.path.join(remote_path, rel_path).replace("\\", "/")
                    file_pairs.append((local_file_path, remote_file_path))

                # if file.endswith('.py'):
                #     local_file_path = os.path.join(root, file)
                #     rel_path = os.path.relpath(local_file_path, local_path)
                #     remote_file_path = os.path.join(remote_path, rel_path).replace("\\", "/")
                #     file_pairs.append((local_file_path, remote_file_path))
    else:
        # 単一ファイル
        remote_file_path = os.path.join(remote_path, os.path.basename(local_path)).replace("\\", "/")
        file_pairs.append((local_path, remote_file_path))

    # 確認ステップ
    if confirm and info:
        print("\n[Upload plan]")
        for i, (local_file, remote_file) in enumerate(file_pairs):
            print(f"{i+1}. {local_file} -> {remote_file}")
        ans = input(f"Total files: {len(file_pairs)}. Proceed with upload? [Y/n]: ").strip().lower()
        if ans not in ['', 'y']:
            print("Upload cancelled.")
            return

    # アップロード実行
    if not file_pairs:
        if info:
            print("No files to upload.")
        return

    github_repo = f"{owner}/{repo}"
    total = len(file_pairs)
    success_count = 0
    start_time = datetime.now()
    for idx, (local_file_path, remote_file_path) in enumerate(file_pairs):
        if info:
            display.progress_bar(idx, total, start_time)
        try:
            upload_github_file(
                github_repo=github_repo,
                branch=branch,
                remote_path=remote_file_path,
                local_file_path=local_file_path,
                commit_message=commit_message,
                github_token=github_token,
                info=info,
                subinfo=subinfo
            )
            success_count += 1
        except Exception as e:
            print(f"\nError uploading {local_file_path}: {e}", file=sys.stderr)
            # エラー発生時も続行
            
    # if info:
    #     display.progress_bar(total, total, start_time, end=True)
    #     print("Upload complete.")

    if info:
        display.current_time_comment(comment=f'Upload complete: {success_count}/{total}')

# def download_github_file(
#     github_url: str,
#     local_base_dir: str = ".",
#     github_token: str | None = None,
#     info: bool = True,
#     subinfo: bool = False
# ):
#     """
#     指定したGitHubファイルURLまたはディレクトリURL（rawでなくてもOK）からファイル/ディレクトリをダウンロードし、ローカルに上書き保存する。
#     プライベートリポジトリの場合はPersonal Access Tokenが必要。

#     Parameters
#     ----------
#     github_url : str
#         GitHub上のファイルまたはディレクトリURL（例: https://github.com/owner/repo/blob/branch/path/to/file.py または https://github.com/owner/repo/tree/branch/path/to/dir）
#     local_base_dir : str
#         ローカルで保存するベースディレクトリ（デフォルト: カレントディレクトリ）
#     github_token : str
#         GitHub Personal Access Token（プライベートリポジトリの場合必須）

#     Returns
#     -------
#     保存したローカルファイルのパスまたはファイルリスト
#     """
#     # ファイルURL
#     m_file = re.match(r"https://github.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)", github_url)
#     # ディレクトリURL
#     m_dir = re.match(r"https://github.com/([^/]+)/([^/]+)/tree/([^/]+)/(.*)", github_url)
#     # ルートディレクトリ（tree/branchのみ）
#     m_dir_root = re.match(r"https://github.com/([^/]+)/([^/]+)/tree/([^/]+)$", github_url)

#     if m_file:
#         owner, repo, branch, file_path = m_file.groups()
#         return _download_github_file_api(owner, repo, branch, file_path, local_base_dir, github_token, info=info, subinfo=subinfo)
#     elif m_dir:
#         owner, repo, branch, dir_path = m_dir.groups()
#         return download_github_dir(owner, repo, branch, dir_path, local_base_dir, github_token, info=info, subinfo=subinfo)
#     elif m_dir_root:
#         owner, repo, branch = m_dir_root.groups()
#         return download_github_dir(owner, repo, branch, '', local_base_dir, github_token, info=info, subinfo=subinfo)
#     else:
#         raise ValueError("URL形式が不正です: " + github_url)






# def remove_pycache_dirs(root_dir):
#     import shutil
#     import os
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         if '__pycache__' in dirnames:
#             pycache_path = os.path.join(dirpath, '__pycache__')
#             print(f"Removing: {pycache_path}")
#             shutil.rmtree(pycache_path)





# def download_github_dir(
#     owner, repo, branch, dir_path, local_base_dir, github_token,
#     confirm: bool = True,
#     file_list: list[str] | None = None,
#     info: bool = True,
#     subinfo: bool = True
# ):
#     """
#     GitHubリポジトリのディレクトリを再帰的にダウンロード
#     """
#     if confirm:
#         file_list = collect_github_files(owner, repo, branch, dir_path, github_token)
#         if info:
#             print(f"\n[Download plan] dir: {dir_path or '.'}")
#             for i, file_path in enumerate(file_list):
#                 local_path = os.path.join(local_base_dir, file_path)
#                 print(f"{i} {file_path} -> {local_path}")
#             ans = input("Download? [Y/n]: ").strip().lower()
#             if ans == 'n':
#                 print("Download cancelled.")
#                 return []
#     # 再帰呼び出し時はfile_listを引き継ぐ
#     if file_list is None:
#         file_list = collect_github_files(owner, repo, branch, dir_path, github_token)

#     api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{dir_path}?ref={branch}"
#     headers = {}
#     if github_token:
#         headers["Authorization"] = f"token {github_token}"
#     r = requests.get(api_url, headers=headers)
#     if r.status_code != 200:
#         print(f"[DEBUG] API URL: {api_url}")
#         print(f"[DEBUG] Headers: {headers}")
#         print(f"[DEBUG] Response: {r.text}")
#         raise Exception(f"GitHub API error: {r.status_code} {r.text}")
#     items = r.json()
#     # __pycache__を除外
#     items = [item for item in items if not (item['type'] == 'dir' and item['name'] == '__pycache__')]
#     downloaded_files = []
#     start_time = datetime.now()
#     for item in items:
#         if item['type'] == 'file':
#             file_path = item['path']
#             idx = file_list.index(file_path)
#             if info:
#                 display.progress_bar(idx, len(file_list), start_time)
#             downloaded_files.append(_download_github_file_api(owner, repo, branch, file_path, local_base_dir, github_token, subinfo=subinfo))
#         elif item['type'] == 'dir':
#             # 再帰的にダウンロード（file_listを引き継ぐ）
#             downloaded_files.extend(download_github_dir(owner, repo, branch, item['path'], local_base_dir, github_token, confirm=False, file_list=file_list, subinfo=subinfo))
#     return downloaded_files


# def download_file_from_github_dir(
#     github_repo: str,
#     branch: str,
#     remote_dir: str,
#     file_name: str,
#     local_base_dir: str = ".",
#     github_token: str | None = None,
#     info: bool = True,
#     subinfo: bool = True
# ):
#     """
#     GitHub上の特定のディレクトリから特定のファイルをローカルにダウンロードする。

#     Parameters
#     ----------
#     github_repo : str
#         GitHubリポジトリ名。例: 'k11kik/Messenger'
#     branch : str
#         ブランチ名。例: 'main'
#     remote_dir : str
#         ダウンロード元のGitHubディレクトリパス。例: 'messenger_analysis/data'
#     file_name : str
#         ダウンロードするファイル名。例: 'data_file.csv'
#     local_base_dir : str
#         ファイルを保存するローカルのベースディレクトリ。
#     github_token : str
#         GitHub Personal Access Token（プライベートリポジトリの場合必須）。
#     info : bool
#         進捗情報を表示するかどうか。
#     subinfo : bool
#         詳細なサブ情報を表示するかどうか。
#     """
#     # GitHub上の完全なリモートパスを構築
#     file_path = os.path.join(remote_dir, file_name).replace("\\", "/")

#     # GitHub APIのURLを構築
#     api_url = f"https://api.github.com/repos/{github_repo}/contents/{file_path}?ref={branch}"
    
#     # download_github_file関数に渡すために、擬似的なURLを作成
#     github_url = f"https://github.com/{github_repo}/blob/{branch}/{file_path}"
    
#     # 既存のdownload_github_file関数を利用してダウンロードを実行
#     return download_github_file(
#         github_url=github_url,
#         local_base_dir=local_base_dir,
#         github_token=github_token,
#         info=info,
#         subinfo=subinfo
#     )


# def upload_all_python_files(
#     local_dir: str,
#     github_repo: str,
#     branch: str,
#     remote_dir: str,
#     commit_message: str,
#     github_token: str,
#     info: bool = True,
#     subinfo: bool = False
# ):
#     """
#     指定したローカルディレクトリ内の全ての.pyファイルをGitHubリポジトリの指定ディレクトリに一括アップロード
#     サブディレクトリも再帰的に対応
#     __pycache__ディレクトリ配下は除外
#     """
#     # 1. 対象ファイル一覧を収集
#     file_pairs = []
#     for root, dirs, files in os.walk(local_dir):
#         if '__pycache__' in dirs:
#             dirs.remove('__pycache__')
#         for file in files:
#             if file.endswith('.py'):
#                 local_file_path = os.path.join(root, file)
#                 rel_path = os.path.relpath(local_file_path, local_dir)
#                 remote_path = os.path.join(remote_dir, rel_path).replace("\\", "/")
#                 file_pairs.append((local_file_path, remote_path))

#     # 2. 一覧表示
#     if info:
#         print(f"\n[Upload plan] {local_dir} -> {github_repo}/{remote_dir}")
#         for i, (local_file_path, remote_path) in enumerate(file_pairs):
#             print(f"{i} {local_file_path} -> {remote_path}")
#         ans = input("Upload? [Y/n]: ").strip().lower()
#         if ans == 'n':
#             print("Upload cancelled.")
#             return

#     # 3. プログレスバー
#     total = len(file_pairs)
#     start_time = datetime.now()
#     for idx, (local_file_path, remote_path) in enumerate(file_pairs):
#         if info:
#             display.progress_bar(idx, total, start_time)
#         upload_github_file(
#             github_repo=github_repo,
#             branch=branch,
#             remote_path=remote_path,
#             local_file_path=local_file_path,
#             commit_message=commit_message,
#             github_token=github_token,
#             info=info,
#             subinfo=subinfo
#         )
