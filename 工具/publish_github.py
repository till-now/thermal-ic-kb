# -*- coding: utf-8 -*-
"""把知识库发布到 GitHub Pages —— 不需要安装 git。

用法：
    python publish_github.py <站点目录> <仓库名> [仓库描述]

令牌来源（按顺序）：
    1. 环境变量 GITHUB_TOKEN
    2. %TEMP%\\gh_token.txt

流程：取用户名 -> 建仓库(或复用) -> 并发上传所有文件为 blob
      -> 建 tree / commit / ref -> 开启 Pages -> 打印访问地址。
上传进度写入 %TEMP%\\gh_blobs.json，中断后重跑会自动跳过已上传的文件。
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://api.github.com"
UA = "kb-publisher/1.0"


def read_token():
    t = os.environ.get("GITHUB_TOKEN")
    if t and t.strip():
        return t.strip()
    p = os.path.join(os.environ.get("TEMP", "."), "gh_token.txt")
    with open(p, encoding="utf-8") as f:
        return f.read().strip()


TOKEN = read_token()
HEADERS = {
    "Authorization": "Bearer " + TOKEN,
    "Accept": "application/vnd.github+json",
    "User-Agent": UA,
    "X-GitHub-Api-Version": "2022-11-28",
}


def api(method, path, payload=None, retries=4):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(API + path, data=data, headers=dict(HEADERS), method=method)
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
                return resp.status, (json.loads(body.decode("utf-8")) if body else None)
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw.decode("utf-8"))
            except Exception:
                return e.code, {"raw": raw[:300].decode("utf-8", "replace")}
        except Exception as e:                       # 网络抖动，退避重试
            last = e
            time.sleep(2 + attempt * 3)
    raise last


def collect(root):
    items = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git",)]
        for fn in filenames:
            if fn == ".kb_output_marker":
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace("\\", "/")
            items.append((rel, full))
    items.sort()
    return items


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    root, repo = sys.argv[1], sys.argv[2]
    desc = sys.argv[3] if len(sys.argv) > 3 else "电力建设标准知识库（DL 5190.4—2019 / DL/T 5210.4—2018）"

    status, me = api("GET", "/user")
    if status != 200:
        print("取用户信息失败:", status, me)
        return 1
    owner = me["login"]
    print("账号:", owner)

    status, _ = api("POST", "/user/repos", {
        "name": repo, "description": desc, "private": False,
        "auto_init": False, "has_issues": False, "has_wiki": False, "has_projects": False,
    })
    if status == 201:
        print("仓库已创建:", repo)
    elif status == 422:
        print("仓库已存在，复用:", repo)
    else:
        print("建仓库失败:", status, _)
        return 1

    files = collect(root)
    total_bytes = sum(os.path.getsize(f) for _, f in files)
    print("待上传 %d 个文件，共 %.1f MB" % (len(files), total_bytes / 1048576))

    # 空仓库不支持 Git Data API（会返回 409 Git Repository is empty），
    # 先用 Contents API 打一个初始提交把仓库“激活”。
    st, res = api("PUT", "/repos/%s/%s/contents/.nojekyll" % (owner, repo),
                  {"message": "初始化仓库", "content": base64.b64encode(b"").decode("ascii")})
    if st in (200, 201):
        print("已创建初始提交")
    elif st == 422:
        print("仓库已有提交")
    else:
        print("初始化仓库返回:", st, res)

    cache_path = os.path.join(os.environ.get("TEMP", "."), "gh_blobs.json")
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception:
            cache = {}

    done = {"n": 0, "bytes": 0}
    t0 = time.time()

    def upload(item):
        rel, full = item
        key = "%s:%d" % (rel, os.path.getsize(full))
        if key in cache:
            return rel, cache[key], os.path.getsize(full), True
        with open(full, "rb") as f:
            content = base64.b64encode(f.read()).decode("ascii")
        st, res = api("POST", "/repos/%s/%s/git/blobs" % (owner, repo),
                      {"content": content, "encoding": "base64"})
        if st != 201:
            raise RuntimeError("blob 失败 %s: %s %s" % (rel, st, res))
        return rel, res["sha"], os.path.getsize(full), False

    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(upload, it): it for it in files}
        for fut in as_completed(futs):
            rel, sha, size, cached = fut.result()
            cache["%s:%d" % (rel, size)] = sha
            results.append((rel, sha))
            done["n"] += 1
            done["bytes"] += size
            if done["n"] % 25 == 0 or done["n"] == len(files):
                el = time.time() - t0
                print("  %d/%d  %.1f/%.1f MB  %.0fs  (%.2f MB/s)" % (
                    done["n"], len(files), done["bytes"] / 1048576, total_bytes / 1048576,
                    el, done["bytes"] / 1048576 / el if el else 0), flush=True)
                json.dump(cache, open(cache_path, "w", encoding="utf-8"))

    json.dump(cache, open(cache_path, "w", encoding="utf-8"))
    print("全部 blob 就绪")

    tree = [{"path": rel, "mode": "100644", "type": "blob", "sha": sha} for rel, sha in results]
    st, res = api("POST", "/repos/%s/%s/git/trees" % (owner, repo), {"tree": tree})
    if st != 201:
        print("建 tree 失败:", st, res)
        return 1
    tree_sha = res["sha"]
    print("tree:", tree_sha)

    st, res = api("POST", "/repos/%s/%s/git/commits" % (owner, repo),
                  {"message": "发布热工仪表及控制装置标准知识库（300 页扫描件 + OCR 全文 + 离线检索）",
                   "tree": tree_sha, "parents": []})
    if st != 201:
        print("建 commit 失败:", st, res)
        return 1
    commit_sha = res["sha"]
    print("commit:", commit_sha)

    st, res = api("POST", "/repos/%s/%s/git/refs" % (owner, repo),
                  {"ref": "refs/heads/main", "sha": commit_sha})
    if st == 201:
        print("分支 main 已建立")
    elif st == 422:
        st2, res2 = api("PATCH", "/repos/%s/%s/git/refs/heads/main" % (owner, repo),
                        {"sha": commit_sha, "force": True})
        print("分支 main 已更新" if st2 == 200 else "更新分支失败: %s %s" % (st2, res2))
    else:
        print("建分支失败:", st, res)
        return 1

    st, res = api("POST", "/repos/%s/%s/pages" % (owner, repo),
                  {"source": {"branch": "main", "path": "/"}})
    if st in (201, 202):
        print("GitHub Pages 已开启")
    elif st == 409:
        print("GitHub Pages 已存在")
    else:
        print("开启 Pages 返回:", st, res)

    url = "https://%s.github.io/%s/" % (owner, repo)
    print("SITE_URL =", url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
