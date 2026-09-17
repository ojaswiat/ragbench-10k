"""Verifies that a rewrite touched only docstrings, never executable code.

Compares the AST of two revisions of the same file with every docstring
replaced by a placeholder. If the dumps differ, something other than a
docstring moved.
"""
import ast
import os
import subprocess
import sys


_DOC_HOLDERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class GitError(Exception):
    pass


class FileNotFoundAtRevision(Exception):
    pass


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if not isinstance(node, _DOC_HOLDERS):
            continue
        body = node.body
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            first.value.value = "<docstring>"
    return tree


def normalise(source: str) -> str:
    return ast.dump(strip_docstrings(ast.parse(source)), annotate_fields=True)


def git_show(rev: str, path: str) -> str:
    """Fetch a file at a given git revision, handling path resolution.

    Resolves the path relative to the repo root, accounting for the current
    working directory. Raises FileNotFoundAtRevision if the file does not
    exist at the revision, and GitError for other failures.
    """
    # Get the repo root
    root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if root_result.returncode != 0:
        raise GitError(f"git rev-parse --show-toplevel failed: {root_result.stderr.strip()}")
    repo_root = root_result.stdout.strip()

    # Get the current directory relative to the repo root
    prefix_result = subprocess.run(
        ["git", "rev-parse", "--show-prefix"],
        capture_output=True,
        text=True,
    )
    if prefix_result.returncode != 0:
        raise GitError(f"git rev-parse --show-prefix failed: {prefix_result.stderr.strip()}")
    prefix = prefix_result.stdout.strip()

    # Normalize the input path: strip leading ./
    normalized_path = path
    if normalized_path.startswith("./"):
        normalized_path = normalized_path[2:]

    # Resolve path relative to repo root
    resolved_path = normalized_path

    if os.path.isabs(normalized_path):
        # Absolute path: strip the repo root to get a repo-relative path
        if normalized_path.startswith(repo_root):
            resolved_path = os.path.relpath(normalized_path, repo_root)
        else:
            # Path is outside the repo
            raise GitError(f"{path} is outside the repository root {repo_root}")
    else:
        # Relative path: prepend the current directory prefix if needed
        if prefix and not normalized_path.startswith(prefix):
            resolved_path = os.path.join(prefix, normalized_path)

    # Check if file exists at the baseline revision
    check_result = subprocess.run(
        ["git", "cat-file", "-e", f"{rev}:{resolved_path}"],
        capture_output=True,
        text=True,
    )

    if check_result.returncode != 0:
        raise FileNotFoundAtRevision(f"{path} does not exist at {rev}")

    # File exists, fetch it
    show_result = subprocess.run(
        ["git", "show", f"{rev}:{resolved_path}"],
        capture_output=True,
        text=True,
    )

    if show_result.returncode != 0:
        raise GitError(f"git show failed for {resolved_path}: {show_result.stderr.strip()}")

    return show_result.stdout


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: check_docstrings_only.py <git-rev> <file> [<file>...]")
        return 2
    rev, paths = argv[1], argv[2:]
    failures = []
    for path in paths:
        try:
            before_content = git_show(rev, path)
        except FileNotFoundAtRevision:
            print(f"NEW  {path}")
            continue
        except GitError as e:
            print(f"ERROR {path}: {e}")
            failures.append(path)
            continue

        try:
            with open(path, encoding="utf-8") as handle:
                after_content = handle.read()
        except Exception as e:
            print(f"ERROR {path}: cannot read current file: {e}")
            failures.append(path)
            continue

        before = normalise(before_content)
        after = normalise(after_content)
        if before != after:
            failures.append(path)
            print(f"FAIL {path}: executable code changed")
        else:
            print(f"ok   {path}")

    if failures:
        print(f"\n{len(failures)} file(s) changed more than docstrings")
        return 1
    print(f"\nall {len(paths)} file(s) docstring-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
