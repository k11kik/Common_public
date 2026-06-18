"""
initial setting
    Put '.github_token.json' in /Users/username/
    >>> {
    "github_token": "your_github_token"
    }
"""


from .github_sync import (
    download,
    upload
)

from .github_token_loader import (
    load_github_token
)