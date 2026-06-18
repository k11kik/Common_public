import os
import sys
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime
from common import display

def parse_version_key(version_str: str) -> tuple:
    """
    バージョン文字列からすべての数値を抽出し、比較用の整数のタプルに変換します。
    これにより、"v2" < "v10" や "v01" < "v2" などの自然な順序での比較を可能にします。
    
    例:
        'v2'     -> (2,)
        'v02'    -> (2,)
        'v10'    -> (10,)
        'v05_01' -> (5, 1)
        'v1.6.2' -> (1, 6, 2)
    """
    if not version_str:
        return (0,)
    # 文字列内の連続する数字の部分をすべて抽出
    digits = re.findall(r'\d+', version_str)
    if not digits:
        return (0,)
    # すべて整数に変換してタプルとして返す
    return tuple(int(d) for d in digits)


def download(
    url: str, 
    output_dir: str = ".", 
    extensions: list | str | None = None,
    username: str | None = None,
    password: str | None = None,
    keywords: list | str | None = None,
    time_format: str | None = r'\d{8}',                # 例: 20110301 (8桁の数字)
    version_format: str | None = r'v\d+(?:_\d+)*|v\d+(?:\.\d+)*', # 例: v05_01 や v1.6.2, v2, v02
    _relative_path: str = ""
):
    """
    URLを指定してファイルをダウンロードする（wget風モジュール）
    再帰的にサブフォルダを巡回し、ローカルに同じディレクトリ構造を構築しながらダウンロードします。
    Basic認証などのID/パスワードによる保護、キーワードフィルタリング、同日ファイルの最新バージョン（数値順）自動選別に対応。
    
    ※実際にダウンロードされるファイルが存在し、通信が成功するまでローカルにディレクトリは作成されません。
    
    Parameters:
    -----------
    url : str
        対象のURL（ファイルへの直リンク、またはディレクトリページ）
    output_dir : str, default "."
        ローカルでの保存先ベースディレクトリ
    extensions : list, str, optional
        フォルダリンクの場合に抽出・ダウンロードする拡張子のリスト（例: ['.cdf', '.png']）。
        Noneの場合は全てのファイルをダウンロードします。
    username : str, optional
        Basic認証等のユーザーID。不要な場合は None。
    password : str, optional
        Basic認証等のパスワード。不要な場合は None。
    keywords : list, str, optional
        ファイル名に含めるべきキーワード、またはキーワードのリスト（例: 'venus'、['rbspa', 'mscb1']）。
        指定されたすべてのキーワードが含まれているファイルのみをダウンロードします。
    time_format : str, optional
        同一日付のグループ化に使用する正規表現パターン。
        デフォルトは 8桁の数字（YYYYMMDDなど）を検出するパターン。
    version_format : str, optional
        バージョン比較に使用する正規表現パターン（例: v05_01 などを抽出）。
        検出されたバージョン文字列が最も大きいもの（数値順）のみをダウンロードします。
    _relative_path : str, internal use only
        再帰呼び出し時に内部的に使用する、起点からの相対ディレクトリパス。
    """
    # サーバーからロボット判定で拒否されるのを防ぐため、一般的なブラウザのUser-Agentを設定
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    # 認証情報を requests 用のオブジェクトに変換 (両方揃っている場合のみ有効化)
    auth = None
    if username is not None and password is not None:
        auth = (username, password)
    
    # 拡張子指定を小文字のリストに統一
    if extensions:
        if isinstance(extensions, str):
            extensions = [extensions.lower()]
        else:
            extensions = [ext.lower() for ext in extensions]

    # キーワードリストを正規化
    if keywords:
        if isinstance(keywords, str):
            keywords = [keywords]
        # 大文字小文字を区別せず比較できるようにすべて小文字に統一して保持
        keywords = [kw.lower() for kw in keywords]

    # 末尾にスラッシュがないディレクトリ指定を補正
    parsed_url = urlparse(url)
    path_lower = parsed_url.path.lower()
    
    # 事前リクエストを送信して Content-Type を取得
    try:
        res = requests.get(url, headers=headers, auth=auth, stream=True, timeout=20)
        res.raise_for_status()
    except Exception as e:
        print(f"[Error] Failed to connect to {url}: {e}", file=sys.stderr)
        return

    content_type = res.headers.get('Content-Type', '')
    
    # フォルダ（ディレクトリインデックスページ）かどうかの判定基準
    is_folder = 'text/html' in content_type and (
        url.endswith('/') or not any(path_lower.endswith(ext) for ext in ['.html', '.htm', '.php', '.jsp', '.aspx'])
    )
    
    # 保存先ローカルフォルダのパスを構成 (ここではフォルダ作成 os.makedirs は行いません)
    target_local_dir = os.path.join(output_dir, _relative_path)
    
    if is_folder:
        if not url.endswith('/'):
            url += '/'
            
        print(f"\n[Directory] Analyzing index: {url}")
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # ページ内のすべてのリンクを抽出
        links = [a.get('href') for a in soup.find_all('a') if a.get('href')]
        
        files_candidate = []
        sub_folders = []
        
        for link in links:
            if link in ['../', './'] or link.startswith('?'):
                continue
                
            abs_url = urljoin(url, link)
            
            if not abs_url.startswith(url):
                continue
                
            if link.endswith('/'):
                sub_folder_name = link.strip('/')
                if sub_folder_name not in sub_folders:
                    sub_folders.append(sub_folder_name)
            else:
                link_path = urlparse(abs_url).path.lower()
                filename = os.path.basename(link_path)
                
                # A. 拡張子フィルタの適用
                if extensions:
                    if not any(link_path.endswith(ext) for ext in extensions):
                        continue
                        
                # B. キーワードフィルタの適用 (AND 条件)
                if keywords:
                    if not all(kw in filename for kw in keywords):
                        continue
                        
                files_candidate.append(abs_url)
                    
        # 重複排除
        files_candidate = list(set(files_candidate))
        
        # C. 同一期間（日付）における最新バージョンのみを選択するロジック
        files_to_download = []
        if files_candidate:
            if time_format and version_format:
                # 日付キーごとのファイル辞書: { date_key: [(version_str, file_url), ...] }
                grouped_by_time = {}
                # パターンにマッチしなかったファイルを格納するリスト
                unmatched_files = []
                
                for file_url in files_candidate:
                    filename = os.path.basename(urlparse(file_url).path)
                    
                    # 日付とバージョンの抽出
                    time_match = re.search(time_format, filename)
                    version_match = re.search(version_format, filename)
                    
                    if time_match and version_match:
                        time_key = time_match.group(0)
                        version_str = version_match.group(0)
                        
                        if time_key not in grouped_by_time:
                            grouped_by_time[time_key] = []
                        grouped_by_time[time_key].append((version_str, file_url))
                    else:
                        unmatched_files.append(file_url)
                
                # 各日付グループにおいて、バージョン文字列が最新（数値的・物理的順序で最大）のものを特定
                for time_key, file_list in grouped_by_time.items():
                    # 文字列ソートではなく、数値を解析したタプルをキーにしてソート
                    file_list.sort(key=lambda x: parse_version_key(x[0]))
                    latest_version, latest_url = file_list[-1]
                    files_to_download.append(latest_url)
                    
                    if len(file_list) > 1:
                        print(f"  [Version Filter] For group '{time_key}', selected latest version '{latest_version}' out of {len(file_list)} candidates.")
                
                # パターンに合致しなかったファイルはそのままダウンロード対象とする
                files_to_download.extend(unmatched_files)
            else:
                files_to_download = files_candidate
        
        # ファイルダウンロード実行
        if files_to_download:
            files_to_download.sort()
            print(f"-> Found {len(files_to_download)} matching files in '{_relative_path or '/'}'")
            start_time_loop = datetime.now()
            for i, file_url in enumerate(files_to_download):
                display.progress_bar(i, len(files_to_download), start_time_loop)
                display.info(f'Accessing: {file_url}')
                _execute_file_download(file_url, target_local_dir, headers, auth)
        else:
            if not sub_folders:
                print(f"-> No files or folders match the criteria in '{_relative_path or '/'}'")

        # B. 検出されたサブフォルダを再帰的にダウンロード (深度優先探索)
        for folder_name in sub_folders:
            sub_url = urljoin(url, folder_name + '/')
            new_rel_path = os.path.join(_relative_path, folder_name)
            print(f"\n[Recursion] Diving into sub-folder: {folder_name}/")
            
            download(
                url=sub_url,
                output_dir=output_dir,
                extensions=extensions,
                username=username,
                password=password,
                keywords=keywords,
                time_format=time_format,
                version_format=version_format,
                _relative_path=new_rel_path
            )
            
    else:
        # 単一ファイルリンクの場合 (キーワード指定があった場合のみ部分一致を確認して処理)
        filename_single = os.path.basename(path_lower)
        should_download = True
        if keywords:
            if not all(kw in filename_single for kw in keywords):
                should_download = False
                print(f"-> Skipping single file '{filename_single}' because it does not match keywords: {keywords}")
                
        if should_download:
            print(f"-> Single file detected.")
            _execute_file_download(url, target_local_dir, headers, auth)


def _execute_file_download(url, target_dir, headers, auth=None):
    """単一のファイルをストリーミングでダウンロードして進捗を表示する内部関数"""
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = "downloaded_file"
        
    filepath = os.path.join(target_dir, filename)
    print(f"  Downloading: {filename} -> {filepath}")
    
    try:
        with requests.get(url, headers=headers, auth=auth, stream=True, timeout=30) as r:
            r.raise_for_status()
            
            # 【重要】通信が完全に成功し、実際に書き込む段階で初めてローカルにディレクトリを作成します
            os.makedirs(target_dir, exist_ok=True)
            
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            block_size = 8192
            
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 簡易プログレスバーの表示
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            status = f"\r     Progress: [{percent:6.1f}%]  {downloaded/(1024*1024):.2f}MB / {total_size/(1024*1024):.2f}MB"
                        else:
                            status = f"\r     Progress: {downloaded/(1024*1024):.2f}MB downloaded"
                        sys.stdout.write(status)
                        sys.stdout.flush()
            print("\n     Finished.")
    except Exception as e:
        print(f"\n     [Failed] Error downloading '{filename}': {e}", file=sys.stderr)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass