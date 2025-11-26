import os
import sys


def pytest_configure():
    # Ensure project root is on sys.path for src imports in smoke tests
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
