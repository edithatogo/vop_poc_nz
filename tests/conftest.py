import os
import sys


def pytest_configure():
    # Ensure src/ layout package is importable during tests
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(repo_root, "src")

    for path in (repo_root, src_path):
        if path not in sys.path:
            sys.path.insert(0, path)
