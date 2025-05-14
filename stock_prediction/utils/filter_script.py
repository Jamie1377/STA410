from git_filter_repo import *
from git_filter_repo import RepoFilter, FilteringArguments

TARGET_COMMIT = b"72c3ae3edef84d87b85c91d7195388badf4b7a89"  # Replace with actual hash


def rewrite_history(commit, metadata):
    if commit.original_id != TARGET_COMMIT:
        # Remove the sensitive file from all commits except the target
        if b"path/to/sensitive-file" in commit.file_changes:
            del commit.file_changes[b"stock_prediction/core/autotrade.py"]
            del commit.file_changes[b"stock_prediction/docs/showcase.ipynb"]
    return commit


args = FilteringArguments.default()
args.force = True
repo_filter = RepoFilter(args, commit_callback=rewrite_history)
repo_filter.run()
