import logging
import sys

# ANSIエスケープコードを定義
ANSI_COLOR_CODES = {
    'DEBUG': '\033[32m',    # 緑
    'INFO': '\033[37m',     # 白/通常
    'WARNING': '\033[33m',  # 黄色
    'ERROR': '\033[31m',    # 赤
    'CRITICAL': '\033[41m\033[37m', # 赤背景、白文字
    'RESET': '\033[0m',     # 色をリセット
}

# ----------------------------------------------------
# 1. カスタムフォーマッタの定義
# ----------------------------------------------------
class CustomColoredFormatter(logging.Formatter):
    """
    指定されたログレベルに応じて、レベル名と日付部分に色を付けるカスタムFormatter。
    
    ご要望のフォーマット: LEVELNAME YYYY-mm-dd HH:MM:SS [func_name] message
    """
    
    # 標準のフォーマット文字列 (色付けは format メソッド内で行う)
    LOG_FORMAT = '%(levelname)s %(asctime)s [%(funcName)s] %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self, fmt=LOG_FORMAT, datefmt=DATE_FORMAT):
        super().__init__(fmt, datefmt)

    def format(self, record):
        """
        ログレコードをフォーマットし、色を適用します。
        
        メッセージ全体ではなく、「LEVELNAME YYYY-mm-dd HH:MM:SS」の部分のみに色を適用します。
        """
        levelname = record.levelname
        
        # ANSIコードを取得
        color_start = ANSI_COLOR_CODES.get(levelname, ANSI_COLOR_CODES['RESET'])
        color_end = ANSI_COLOR_CODES['RESET']
        
        # 1. タイムスタンプを整形
        time_str = self.formatTime(record, self.datefmt)
        
        # 2. フォーマット文字列全体を構築
        # LEVELNAME TIME [funcName] message
        
        # 【キーポイント】
        # レベル名とタイムスタンプのみに色を適用するため、
        # まず色なしで全体の文字列を生成する
        
        # 一時的にFormatterのフォーマットを変更して、基底クラスの format() を呼び出す
        
        # 呼び出し元の関数名を取得。これが空の場合を考慮
        func_name_part = f"[{record.funcName}]" if record.funcName else ""
        
        # 色付けの対象となる接頭辞を生成
        prefix_str = f"{record.levelname} {time_str}"

        # ログメッセージの残りの部分を生成 (funcName と message)
        # 基底クラスの format を呼び出すため、一時的なFormatterを作成し、
        # 最終的なメッセージ部分のみを取得します
        
        # NOTE: self.format() を呼び出すと再帰になるため、logging.Formatter を利用
        # ただし、最も安全なのは、必要なフィールドを record から直接取得して組み立てる方法です。
        
        message_part = f"{func_name_part} {record.getMessage()}"
        
        # 最終的な色付きのログ行を組み立てる
        # (色開始)LEVELNAME TIME(色終了) [funcName] message
        final_log = f"{color_start}{prefix_str}{color_end} {message_part}"
        
        return final_log.strip()


# ----------------------------------------------------
# 2. ロガー設定関数
# ----------------------------------------------------

def setup_colored_logging(logger_name=None, level=logging.DEBUG):
    """
    指定されたロガー、またはルートロガーにカスタムカラー設定を適用します。
    """
    logger = logging.getLogger(logger_name)
    
    # 【最重要】ロガーのレベルを DEBUG に設定し、DEBUG メッセージが出力されるようにする
    logger.setLevel(level) 
    
    # 【最重要】既存のハンドラーをすべてクリアする
    # これが、意図しないデフォルトのフォーマットが表示される主要因です。
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    
    # 標準出力 (コンソール) 用のハンドラーを作成
    stream_handler = logging.StreamHandler(sys.stdout)
    
    # ハンドラーのレベルも DEBUG に設定する
    stream_handler.setLevel(level)
    
    # カスタムフォーマッタをインスタンス化
    formatter = CustomColoredFormatter()
    
    # ハンドラーにカスタムフォーマッタを設定
    stream_handler.setFormatter(formatter)
    
    # ロガーにハンドラーを追加
    logger.addHandler(stream_handler)

# ----------------------------------------------------
# 3. モジュールがインポートされたときにデフォルト設定を適用
# ----------------------------------------------------
# このファイルを import するだけで、ルートロガーに設定が適用されます。
# if __name__ == '__main__':
#     setup_colored_logging()
