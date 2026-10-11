# -*- coding: utf-8 -*-
"""构建《热工仪表及控制装置》个人知识库。

输入：两份 DL 系列标准的扫描版 PDF。每页为 1 张扫描图 + 隐形 OCR 文本层，
      且多数页面叠加了**两层 OCR**（较早的一层准确，后一层严重乱码）。
处理：逐页定位第二层 OCR 的起点，用字频 + 二元组语言模型选出可用文本；
      按书签切章、抽取条文号索引、无损导出页面原图、生成离线检索页。
"""
import collections
import csv
import io
import json
import math
import os
import re
import shutil

import numpy as np
import pymupdf
from PIL import Image

WS = r"C:\Users\田浩\Documents\deepseek-harness\default-workspace"
OUT = os.path.join(WS, "热工仪表及控制装置知识库")
MARKER = ".kb_output_marker"

SPECS = [
    dict(
        key="5190.4",
        code="DL 5190.4—2019",
        title="电力建设施工技术规范 第4部分：热工仪表及控制装置",
        folder="01_DL 5190.4-2019 施工技术规范_热工仪表及控制装置",
        src=r"E:\XXX项目部_模板\05规范图集\标准图集、手册\5190规范\5190.4-2019《电力建设施工技术规范 第4部分：热工仪表及控制装置》书签版.pdf",
        rotate_cw=90,
    ),
    dict(
        key="5210.4",
        code="DL/T 5210.4—2018",
        title="电力建设施工质量验收规程 第4部分：热工仪表及控制装置",
        folder="02_DL T 5210.4-2018 质量验收规程_热工仪表及控制装置",
        src=r"E:\XXX项目部_模板\05规范图集\标准图集、手册\5210规范\5210.4-2018电力建设施工质量验收规程 第4部分：热工仪表及控制装置 书签版.pdf",
        rotate_cw=270,
    ),
]

CLAUSE_FULL = re.compile(r"(?m)^(\d{1,2}(?:\.\d{1,3}){1,3})(?=[ \t]*[\u4e00-\u9fff])")
CLAUSE_GLUED = re.compile(r"(?<![\d.\-])(\d{1,2}(?:\.\d{1,3}){1,3})(?=\s*[\u4e00-\u9fff])")
MARK_NUM = re.compile(r"(?:5190|5210)")
CJK = re.compile(r"[\u4e00-\u9fff]")


# --------------------------------------------------------------------------
# 文本层判别
# --------------------------------------------------------------------------
def blocks_of(page):
    return [b for b in page.get_text("blocks") if int(b[6]) == 0 and b[4].strip()]


def bbox_close(a, b, tol=14.0):
    return all(abs(a[i] - b[i]) <= tol for i in range(4))


def first_dup_split(blocks, tol=14.0):
    """第二层 OCR 会重复第一层文本块的位置；按块中心距找出第一个重复块。"""
    n = len(blocks)
    for k in range(2, n):
        bx = blocks[k][:4]
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        for j in range(0, k - 2):
            ax = blocks[j][:4]
            if abs(cx - (ax[0] + ax[2]) / 2) <= tol and abs(cy - (ax[1] + ax[3]) / 2) <= tol:
                return k
    return None


def is_running_head(ln, key):
    if len(ln) > 34 or CJK.search(ln):
        return False
    s = ln.replace(" ", "")
    if key == "5190.4":
        return "5190" in s and ("2015" in s or "2019" in s)
    return "5210" in s and "2018" in s


class LangModel:
    """以 5190.4（文本质量最好的那本）为语料的一元 + 二元语言模型。"""

    def __init__(self, corpus):
        corpus = re.sub(r"\s+", "", corpus)
        self.uni = collections.Counter(corpus)
        self.bi = collections.Counter(zip(corpus, corpus[1:]))
        self.tu = sum(self.uni.values()) or 1
        self.tb = sum(self.bi.values()) or 1
        self.a = 0.01

    def __call__(self, t):
        t = re.sub(r"\s+", "", t)
        if len(t) < 6:
            return -99.0
        su = sum(math.log((self.uni.get(c, 0) + self.a) / self.tu) for c in t) / len(t)
        sb = sum(math.log((self.bi.get(p, 0) + self.a) / self.tb)
                 for p in zip(t, t[1:])) / max(1, len(t) - 1)
        return su + sb


def block_is_head(btext, key):
    for ln in btext.split("\n"):
        if ln.strip() and is_running_head(ln.strip(), key):
            return True
    return False


def pick_text(blocks, spec, model=None):
    """返回 (该页可用文本, 判别结果标签)。

    两层 OCR 在内容流中前后相接：先是一层较准的，随后是整层乱码。
    分界靠两个信号——第二层会重复第一层的文本块位置，也会重新出现页眉；
    默认保留前一层；只有当前层只是一小段残片、且语言模型明确支持后层时才取后层。
    """
    n = len(blocks)
    if n < 4:
        return "".join(b[4] for b in blocks), "all"

    kd = first_dup_split(blocks)
    kh = None
    for i in range(1, n):
        if block_is_head(blocks[i][4], spec["key"]):
            kh = i
            break

    if kd is None and kh is None:
        return "".join(b[4] for b in blocks), "single"

    # 有位置重复证据时以它为准；否则退而用页眉重现行
    k = kd if kd is not None else kh

    bounds = [0]
    for b in blocks:
        bounds.append(bounds[-1] + len(b[4]))
    full = "".join(b[4] for b in blocks)
    pre, suf = full[:bounds[k]], full[bounds[k]:]
    total = bounds[-1] or 1

    if kd is None:
        # 只有页眉一个信号，且切口太靠前，不足以判定为两层 —— 宁可不切
        if len(pre) < 0.25 * total:
            return "".join(b[4] for b in blocks), "single"
        return pre, "layerA"

    # 少数页面前层是旋转乱码、后层才是正文：后层显著更好且前层占比不大时改用后层
    if (model is not None and len(pre) >= 6 and len(suf) >= 6
            and len(pre) < 0.7 * total and model(suf) > model(pre) + 2.0):
        return suf, "layerB"
    return pre, "layerA"


def tidy(raw, spec):
    lines, printed = [], None
    for ln in raw.split("\n"):
        ln = ln.replace("\u3000", " ").strip()
        if not ln or is_running_head(ln, spec["key"]):
            continue
        lines.append(ln)
    printed = None
    if len("\n".join(lines)) > 120:
        if lines and re.fullmatch(r"\d{1,4}", lines[-1]):
            printed = lines.pop()
        if lines and re.fullmatch(r"\d{1,3}", lines[0]):
            printed = printed or lines.pop(0)
    text = "\n".join(lines)
    text = CLAUSE_GLUED.sub(lambda m: "\n" + m.group(1) + " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), printed


def extract_text(doc, spec, model=None):
    out = []
    for pno in range(doc.page_count):
        blocks = blocks_of(doc[pno])
        raw, kind = pick_text(blocks, spec, model)
        text, printed = tidy(raw, spec)
        out.append(dict(no=pno + 1, text=text, kind=kind, printed=printed))
    return out


def review_flags(doc, spec, pages, model):
    """列出“语言模型认为被丢弃的那一层更好”的页，供人工复核（保留前层不变）。"""
    flags = []
    for p in pages:
        if p["kind"] != "layerA":
            continue
        blocks = blocks_of(doc[p["no"] - 1])
        kd = first_dup_split(blocks)
        kh = None
        for i in range(1, len(blocks)):
            if block_is_head(blocks[i][4], spec["key"]):
                kh = i
                break
        if kd is None and kh is None:
            continue
        k = kd if kd is not None else kh
        bounds = [0]
        for b in blocks:
            bounds.append(bounds[-1] + len(b[4]))
        full = "".join(b[4] for b in blocks)
        total = bounds[-1] or 1
        if kd is None and bounds[k] < 0.25 * total:
            continue
        pre, suf = full[:bounds[k]], full[bounds[k]:]
        if len(pre) < 6 or len(suf) < 6:
            continue
        d = model(suf) - model(pre)
        if d > 0.8:
            flags.append((p["no"], round(d, 2)))
    return sorted(flags, key=lambda x: -x[1])


# --------------------------------------------------------------------------
def safe_name(s, maxlen=60):
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", s).strip(" .")
    s = re.sub(r"\s+", " ", s)
    return s[:maxlen] or "未命名"


def prepare_out():
    if os.path.isdir(OUT):
        if not (os.path.exists(os.path.join(OUT, MARKER)) or not os.listdir(OUT)):
            raise SystemExit("输出目录已存在且非本工具创建，拒绝覆盖：%s" % OUT)
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    with open(os.path.join(OUT, MARKER), "w", encoding="utf-8") as f:
        f.write("本目录由 工具/build_kb.py 生成，可整体删除后重建\n")


def chapter_ranges(doc):
    toc = doc.get_toc(simple=True)
    n = doc.page_count
    entries = []
    for i, (lvl, title, p) in enumerate(toc):
        end = n
        for j in range(i + 1, len(toc)):
            if toc[j][2] > p:
                end = toc[j][2] - 1
                break
        entries.append(dict(level=lvl, title=title, start=p, end=max(p, end)))
    front = dict(level=0, title="封面 · 公告 · 前言 · 目次", start=0, end=0)
    return entries, front


def main():
    # 本脚本可能就放在输出目录里，先把它读进内存，避免被 prepare_out 清掉
    self_src = None
    try:
        with open(os.path.abspath(__file__), "r", encoding="utf-8") as f:
            self_src = f.read()
    except Exception:
        pass

    prepare_out()
    overview = os.path.join(OUT, "00_总览")
    search_dir = os.path.join(OUT, "检索")
    for d in (overview, search_dir, os.path.join(OUT, "工具")):
        os.makedirs(d, exist_ok=True)

    # ---- 第 1 遍：以“保留较早一层”的朴素规则取 5190.4 文本，作为语言模型语料 ----
    spec0 = SPECS[0]
    doc0 = pymupdf.open(spec0["src"])
    corpus = "".join(p["text"] for p in extract_text(doc0, spec0, None))
    doc0.close()
    model = LangModel(corpus)
    print("语料：%d 字，%d 个不同字，%d 个不同二元组" % (
        model.tu, len(model.uni), len(model.bi)))

    all_pages, clause_rows, page_rows, stats = [], [], [], {}
    toc_md = [
        "# 目录树",
        "",
        "两份标准均为**扫描件 + 隐形 OCR 文本层**（多数页面叠加了两层 OCR，已自动择优选层）。",
        "表中页码为 **PDF 页码**；原书印刷页码见 `页面清单.csv` 的“印刷页码”列。",
        "",
        "| 标准 | 起止 | 分章全文 | 原图 |",
        "| --- | --- | --- | --- |",
    ]

    for si, spec in enumerate(SPECS):
        print("== 处理 %s" % spec["code"])
        doc = pymupdf.open(spec["src"])
        n_pages = doc.page_count
        base = os.path.join(OUT, spec["folder"])
        img_dir, chap_dir = os.path.join(base, "原图"), os.path.join(base, "分章")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(chap_dir, exist_ok=True)

        entries, front = chapter_ranges(doc)
        pages = extract_text(doc, spec, model)
        kinds = collections.Counter(p["kind"] for p in pages)
        flags = review_flags(doc, spec, pages, model)

        def chapter_of(pno):
            p = pno + 1
            cur = None
            for e in entries:
                if e["start"] <= p:
                    cur = e
                else:
                    break
            return cur or front

        # ---- 原图 + 水印检测 ----
        img_bytes = 0
        watermarks = []
        for pno in range(n_pages):
            pg = doc[pno]
            try:
                imgs = pg.get_images(full=True)
                data = doc.extract_image(imgs[0][0])["image"] if imgs else None
                if data:
                    im = Image.open(io.BytesIO(data))
                else:
                    pix = pg.get_pixmap(dpi=200)
                    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                if im.mode not in ("L", "RGB"):
                    im = im.convert("L" if im.mode in ("1", "L", "LA") else "RGB")
                ang = spec["rotate_cw"] % 360
                if ang:
                    im = im.rotate(-ang, expand=True)
                fp = os.path.join(img_dir, "p%03d.jpg" % (pno + 1))
                im.save(fp, "JPEG", quality=88, optimize=True)
                img_bytes += os.path.getsize(fp)
            except Exception as e:  # noqa: BLE001
                print("  ! 原图失败 p%d: %r" % (pno + 1, e))
            try:
                pix = pg.get_pixmap(dpi=36)
                a = np.frombuffer(pix.samples, dtype=np.uint8)
                if pix.n >= 3:
                    a = a.reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(np.int16)
                    m = (a[:, :, 0] > 195) & (a[:, :, 1] > 110) & (a[:, :, 1] < 205) & (a[:, :, 2] < 135)
                    if m.mean() > 0.004:
                        watermarks.append(pno + 1)
            except Exception:
                pass

        # ---- 分章 / 全文 ----
        full_lines = [
            "# %s %s" % (spec["code"], spec["title"]),
            "",
            "> 来源 PDF：`%s`" % os.path.basename(spec["src"]),
            "> 共 %d 页。页码为 PDF 页码。" % n_pages,
            "",
            "[返回目录树](../00_总览/目录树.md) ｜ [条款索引](../00_总览/条文索引.csv) ｜ [离线检索](../检索/index.html)",
            "",
        ]
        groups = collections.OrderedDict()
        for p in pages:
            groups.setdefault((chapter_of(p["no"] - 1)["start"], chapter_of(p["no"] - 1)["title"]),
                              []).append(p)

        toc_md.append("| %s | %d 页 | [全文.md](%s/全文.md) | [原图/](%s/原图/) |" % (
            spec["code"], n_pages, spec["folder"], spec["folder"]))
        toc_md += ["", "### %s %s" % (spec["code"], spec["title"]), "",
                   "| 章节 | PDF 页码 | 分章全文 |", "| --- | --- | --- |"]

        chap_index = 0
        for (start, title), plist in groups.items():
            fname = "%03d_%s.md" % (chap_index, safe_name(title))
            chap_index += 1
            p_from, p_to = plist[0]["no"], plist[-1]["no"]
            body = ["# %s" % title, "",
                    "> %s ｜ PDF 第 %d–%d 页 ｜ [返回目录树](../00_总览/目录树.md)" % (
                        spec["code"], p_from, p_to), ""]
            for p in plist:
                body += ['<a id="p%d"></a>' % p["no"], "", "## 第 %d 页" % p["no"], "",
                         "[查看原图](../原图/p%03d.jpg)" % p["no"], "",
                         p["text"] if p["text"] else "*（本页无可提取文本，请查看原图）*", ""]
            with open(os.path.join(chap_dir, fname), "w", encoding="utf-8") as f:
                f.write("\n".join(body))
            toc_md.append("| %s | %d–%d | [%s](../%s/分章/%s) |" % (
                title, p_from, p_to, title, spec["folder"], fname))
            full_lines += ["", "## %s（PDF 第 %d–%d 页）" % (title, p_from, p_to), ""]
            for p in plist:
                full_lines += ["### 第 %d 页" % p["no"], "",
                               p["text"] if p["text"] else "*（本页无可提取文本）*", ""]
        toc_md.append("")

        with open(os.path.join(base, "全文.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(full_lines))

        # ---- 条款索引 / 页面清单 ----
        seen = set()
        for p in pages:
            ch = chapter_of(p["no"] - 1)["title"]
            for m in CLAUSE_FULL.finditer(p["text"]):
                num = m.group(1)
                if num.count(".") > 3 or int(num.split(".")[0]) > 20:
                    continue
                if (spec["key"], num) in seen:
                    continue
                seen.add((spec["key"], num))
                clause_rows.append([spec["code"], num, ch, p["no"],
                                    p["text"][m.end():m.end() + 34].replace("\n", " ").strip()])
        for p in pages:
            ch = chapter_of(p["no"] - 1)["title"]
            page_rows.append([spec["code"], p["no"], p["printed"] or "", ch, len(p["text"]),
                              "是" if re.search(r"表\s*\d+(?:\.\d+)*", p["text"]) else "",
                              "是" if p["no"] in watermarks else "",
                              "是" if not p["text"] else "",
                              "p%03d.jpg" % p["no"]])
            all_pages.append(dict(s=si, p=p["no"], t=ch, x=p["text"]))

        stats[spec["key"]] = dict(
            code=spec["code"], title=spec["title"], pages=n_pages, chapters=len(groups),
            clauses=len(seen), chars=sum(len(p["text"]) for p in pages),
            img_mb=round(img_bytes / 1048576.0, 1), layers=dict(kinds),
            watermark_pages=watermarks,
            review_pages=[f[0] for f in flags],
            empty_pages=[p["no"] for p in pages if not p["text"]])
        doc.close()

    # ---- 索引输出 ----
    with open(os.path.join(overview, "条文索引.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["标准", "条文号", "所属章节", "PDF页码", "原文片段"])
        w.writerows(clause_rows)
    with open(os.path.join(overview, "条文索引.json"), "w", encoding="utf-8") as f:
        json.dump([dict(zip(["标准", "条文号", "所属章节", "PDF页码", "原文片段"], r))
                   for r in clause_rows], f, ensure_ascii=False, indent=1)

    with open(os.path.join(overview, "页面清单.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["标准", "PDF页码", "印刷页码", "所属章节", "提取字数",
                    "含表格", "有水印", "无文本", "原图文件"])
        w.writerows(page_rows)

    with open(os.path.join(overview, "目录树.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(toc_md))

    payload = dict(stds=[dict(code=s["code"], title=s["title"], folder=s["folder"]) for s in SPECS],
                   pages=all_pages)
    with open(os.path.join(search_dir, "_data.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    with open(os.path.join(search_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(HTML.replace("__PAYLOAD__",
                             json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")))
    with open(os.path.join(overview, "_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    write_readme(stats)
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(ROOT_INDEX)
    if self_src:
        with open(os.path.join(OUT, "工具", "build_kb.py"), "w", encoding="utf-8") as f:
            f.write(self_src)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


ROOT_INDEX = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>热工仪表及控制装置 · 标准知识库</title>
<meta http-equiv="refresh" content="0;url=%E6%A3%80%E7%B4%A2/index.html">
<style>
body{margin:0;padding:52px 22px;font:16px/1.9 "Microsoft YaHei","Segoe UI",sans-serif;text-align:center;color:#1b1f24}
a{color:#0b62d0}
</style>
</head>
<body>
<p>正在进入「热工仪表及控制装置」标准知识库…</p>
<p><a href="%E6%A3%80%E7%B4%A2/index.html">如果没有自动跳转，点这里</a></p>
</body>
</html>
"""


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>热工仪表及控制装置 · 标准知识库</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--line:#e3e6ea;--fg:#1b1f24;--mut:#6b7480;--acc:#0b62d0;--hl:#ffe680}
*{box-sizing:border-box}
body{margin:0;font:15px/1.7 "Microsoft YaHei","Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:12px 18px}
h1{margin:0 0 8px;font-size:18px;font-weight:700}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
input[type=search]{flex:1;min-width:240px;padding:9px 12px;font-size:15px;border:1px solid var(--line);border-radius:8px}
select,button{padding:8px 12px;font-size:14px;border:1px solid var(--line);border-radius:8px;background:#fff}
button{cursor:pointer}
button.acc{background:var(--acc);color:#fff;border-color:var(--acc)}
.meta{color:var(--mut);font-size:13px;margin-top:6px}
main{display:grid;grid-template-columns:minmax(320px,430px) 1fr;gap:14px;padding:14px 18px;align-items:start}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px}
#list{max-height:calc(100vh - 168px);overflow:auto}
.hit{padding:10px 12px;border-bottom:1px solid var(--line);cursor:pointer}
.hit:hover{background:#f0f5ff}
.hit.on{background:#e6f0ff}
.hit .t{font-size:13px;color:var(--acc);font-weight:600}
.hit .p{font-size:12px;color:var(--mut);margin:2px 0 4px}
.hit .s{font-size:13px;color:#333;word-break:break-all}
mark{background:var(--hl);padding:0 1px}
#detail{padding:14px 18px;max-height:calc(100vh - 168px);overflow:auto}
#detail h2{margin:0 0 4px;font-size:16px}
#detail .sub{color:var(--mut);font-size:13px;margin-bottom:10px}
#detail pre{white-space:pre-wrap;word-break:break-word;font:14px/1.85 "Microsoft YaHei",monospace;margin:0}
#detail img{max-width:100%;border:1px solid var(--line);border-radius:6px;margin-top:12px}
.empty{color:var(--mut);padding:18px}
.tabs{display:flex;gap:8px;margin-top:8px}
.tabs button.on{background:#e6f0ff;border-color:var(--acc);color:var(--acc)}
#toc{padding:12px 16px;max-height:calc(100vh - 208px);overflow:auto;display:none}
#toc .st{font-weight:600;margin:12px 0 6px;color:var(--acc)}
#toc a{display:block;padding:3px 0;color:var(--fg);text-decoration:none;font-size:14px}
#toc a:hover{color:var(--acc);text-decoration:underline}
</style>
</head>
<body>
<header>
  <h1>热工仪表及控制装置 · 标准知识库 <span class="meta" id="scope"></span></h1>
  <div class="row">
    <input type="search" id="q" placeholder="全文检索：取源部件 / 屏蔽接地 / 仪表校验 / 主控 / 电缆敷设" autofocus>
    <select id="std"><option value="-1">全部标准</option></select>
    <button class="acc" id="go">检索</button>
    <button id="clr">清空</button>
  </div>
  <div class="tabs">
    <button id="tabSearch" class="on">检索结果</button>
    <button id="tabToc">章节目录</button>
  </div>
  <div class="meta" id="stat">文本由扫描件 OCR 得到，正式引用请以原图为准。</div>
</header>
<main>
  <div class="card">
    <div id="list"><div class="empty">在上方输入关键词开始检索。</div></div>
    <div id="toc"></div>
  </div>
  <div class="card" id="detail"><div class="empty">左侧选择一条结果，这里显示该页全文与扫描原图。</div></div>
</main>
<script>
const DATA = __PAYLOAD__;
const stds = DATA.stds, pages = DATA.pages;
const $ = s => document.querySelector(s);
const esc = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const imgPath = pg => '../' + stds[pg.s].folder + '/原图/p' + String(pg.p).padStart(3,'0') + '.jpg';

function init(){
  const sel = $('#std');
  stds.forEach((s,i)=>{ const o=document.createElement('option'); o.value=i; o.textContent=s.code; sel.appendChild(o); });
  const chars = pages.reduce((a,p)=>a+p.x.length,0);
  $('#scope').textContent = '（'+stds.length+' 份标准 / '+pages.length+' 页 / 约 '+(chars/10000).toFixed(1)+' 万字）';
  buildToc();
}
function buildToc(){
  const box = $('#toc'), map = new Map();
  let html = '';
  pages.forEach(p=>{ const k=p.s+'|'+p.t;
    if(!map.has(k)) map.set(k,{s:p.s,t:p.t,a:p.p,b:p.p}); else map.get(k).b=p.p; });
  let cur = null;
  map.forEach(v=>{
    if(v.s!==cur){ cur=v.s; html += '<div class="st">'+esc(stds[v.s].code+' '+stds[v.s].title)+'</div>'; }
    html += '<a href="#" data-s="'+v.s+'" data-p="'+v.a+'">'+esc(v.t)+
            ' <span style="color:#6b7480">（PDF 第 '+v.a+(v.b>v.a?'–'+v.b:'')+' 页）</span></a>';
  });
  box.innerHTML = html;
  box.querySelectorAll('a').forEach(a=>a.onclick=e=>{
    e.preventDefault(); show(pages.find(p=>p.s==a.dataset.s && p.p==a.dataset.p)); });
}
function snippet(text,q){
  const i = text.toLowerCase().indexOf(q.toLowerCase());
  if(i<0) return esc(text.slice(0,160));
  const a=Math.max(0,i-60), b=Math.min(text.length,i+q.length+120);
  return (a>0?'…':'')+esc(text.slice(a,i))+'<mark>'+esc(text.slice(i,i+q.length))+'</mark>'+
         esc(text.slice(i+q.length,b))+(b<text.length?'…':'');
}
function search(){
  const q=$('#q').value.trim(), s=parseInt($('#std').value,10), box=$('#list');
  if(!q){ box.innerHTML='<div class="empty">请输入关键词。</div>'; return; }
  const ql=q.toLowerCase(), hits=[];
  for(const p of pages){
    if(s>=0 && p.s!==s) continue;
    const xl=p.x.toLowerCase();
    let idx=xl.indexOf(ql), cnt=0;
    while(idx>=0 && cnt<50){ cnt++; idx=xl.indexOf(ql, idx+ql.length); }
    if(cnt) hits.push({p,cnt});
  }
  if(!hits.length){ box.innerHTML='<div class="empty">没有匹配结果，可尝试更短的关键词。</div>'; $('#stat').textContent=''; return; }
  $('#stat').textContent='命中 '+hits.length+' 页 / 共 '+hits.reduce((a,h)=>a+h.cnt,0)+' 处（点击查看该页全文与原图）';
  box.innerHTML = hits.map((h,i)=>
    '<div class="hit" data-i="'+i+'"><div class="t">'+esc(stds[h.p.s].code)+'</div>'+
    '<div class="p">PDF 第 '+h.p.p+' 页 ｜ '+esc(h.p.t)+' ｜ '+h.cnt+' 处</div>'+
    '<div class="s">'+snippet(h.p.x,q)+'</div></div>').join('');
  box.querySelectorAll('.hit').forEach(el=>el.onclick=()=>{
    box.querySelectorAll('.hit').forEach(x=>x.classList.remove('on'));
    el.classList.add('on'); show(hits[+el.dataset.i].p, q);
    if(window.innerWidth<=900) $('#detail').scrollIntoView({behavior:'smooth',block:'start'}); });
}
function show(p,q){
  q = q || $('#q').value.trim();
  let body;
  if(q){
    const parts = p.x.split(new RegExp('('+q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')','gi'));
    body = parts.map((s,i)=> i%2 ? '<mark>'+esc(s)+'</mark>' : esc(s)).join('');
  } else body = esc(p.x);
  $('#detail').innerHTML = '<style>.back{display:none;margin:0 0 10px}@media(max-width:900px){.back{display:block}}</style>'+
    '<div class="back"><a href="#" id="backBtn">← 返回检索结果</a></div>'+
    '<h2>'+esc(p.t)+'</h2>'+
    '<div class="sub">'+esc(stds[p.s].code+' '+stds[p.s].title)+' ｜ PDF 第 '+p.p+
    ' 页 ｜ <a href="'+imgPath(p)+'" target="_blank">单独打开原图</a></div>'+
    '<pre>'+body+'</pre><img loading="lazy" src="'+imgPath(p)+'" alt="第'+p.p+'页原图">';
  const bb = document.getElementById('backBtn');
  if(bb) bb.onclick = e => { e.preventDefault();
    $('#list').scrollIntoView({behavior:'smooth',block:'start'}); };
  $('#detail').scrollTop = 0;
}
$('#go').onclick = search;
$('#q').addEventListener('keydown', e=>{ if(e.key==='Enter') search(); });
$('#q').addEventListener('input', ()=>{ if($('#q').value.trim().length>=2) search(); });
$('#std').onchange = ()=>{ if($('#q').value.trim()) search(); };
$('#clr').onclick = ()=>{ $('#q').value='';
  $('#list').innerHTML='<div class="empty">在上方输入关键词开始检索。</div>'; $('#stat').textContent=''; };
$('#tabSearch').onclick = ()=>{ $('#tabSearch').classList.add('on'); $('#tabToc').classList.remove('on');
  $('#list').style.display=''; $('#toc').style.display='none'; };
$('#tabToc').onclick = ()=>{ $('#tabToc').classList.add('on'); $('#tabSearch').classList.remove('on');
  $('#list').style.display='none'; $('#toc').style.display='block'; };
init();
</script>
</body>
</html>
"""


def write_readme(stats):
    L = [
        "# 热工仪表及控制装置 · 个人知识库",
        "",
        "由两份电力行业标准的**书签版扫描 PDF** 自动提取、结构化而成，完全离线可用。",
        "",
        "## 收录标准",
        "",
        "| 标准 | 名称 | 页数 | 分章 | 条款条目 | 提取字数 | 原图 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for v in stats.values():
        L.append("| %s | %s | %d | %d | %d | %s | %d 张 / %s MB |" % (
            v["code"], v["title"], v["pages"], v["chapters"], v["clauses"],
            format(v["chars"], ","), v["pages"], v["img_mb"]))
    L += [
        "",
        "## 目录结构",
        "",
        "```",
        "热工仪表及控制装置知识库/",
        "├─ README.md                     本文件",
        "├─ 00_总览/",
        "│   ├─ 目录树.md                  两份标准的完整章节树",
        "│   ├─ 条文索引.csv / .json        条文号 → 章节 → PDF 页码 → 原文片段",
        "│   ├─ 页面清单.csv               逐页：印刷页码、章节、字数、是否有表格/水印",
        "│   └─ _stats.json               构建统计",
        "├─ 01_DL 5190.4-2019 施工技术规范_热工仪表及控制装置/",
        "│   ├─ 全文.md                   全书连续全文（按页标注）",
        "│   ├─ 分章/*.md                 每章一个文件",
        "│   └─ 原图/pNNN.jpg             逐页扫描原图（已校正方向）",
        "├─ 02_DL T 5210.4-2018 质量验收规程_热工仪表及控制装置/   同上",
        "├─ 检索/index.html               【双击打开】离线全文检索界面",
        "└─ 工具/build_kb.py              生成脚本",
        "```",
        "",
        "## 怎么用",
        "",
        "1. **全文检索**：双击 `检索/index.html`（浏览器打开，不联网、不需服务器）。",
        "   输入关键词即时出结果；点左侧条目，右侧显示该页全文 + 扫描原图。",
        "2. **按章阅读**：从 `00_总览/目录树.md` 进入，或直接看某本标准的 `全文.md`。",
        "3. **按条文号定位**：在 `00_总览/条文索引.csv` 查条文号（如 `4.1.11`）得到 PDF 页码，",
        "   再打开 `原图/pNNN.jpg` 核对。",
        "4. **表格类内容**：5210.4 的验收表、记录表以**原图为准**（见下）。",
        "",
        "## 构建方法",
        "",
        "两份 PDF 都是**扫描图像 + 隐形 OCR 文本层**，且多数页面被叠加了**两层 OCR**",
        "（较早的一层准确，后一层严重乱码，两者在页面上的位置几乎重合）。处理链：",
        "",
        "1. 逐页取文本块，按“块中心位置重复”定位第二层 OCR 的起点，辅以页眉重复出现的位置；",
        "2. 以 5190.4 的文本训练一元字频 + 二元组语言模型，用于校验选层结果",
        "   （能区分“敷设/数设”“电缆/电统”这类同字频错误），并识别“前层是旋转乱码”的少数页面；",
        "3. 剔除页眉（标准号）与页脚孤立页码，按条文号断行，生成可读全文；",
        "4. 按 PDF 书签切分章节；正则抽取条文号形成索引；",
        "5. 抽取每页内嵌扫描图（原始像素，无损），按页面 `/Rotate` 校正方向后存 JPEG；",
        "6. 生成自包含的单文件离线检索页（全文内嵌、图片相对路径）。",
        "",
        "## 已知问题（重要）",
        "",
        "- **文本来自扫描件 OCR，存在识别错误**（字词、数字都可能错，例如 `4.1.11` 可能被识别成 `4.1.1`）。",
        "  用于检索、定位、通读没问题；凡作为**施工依据、验收判定、对外报审**，必须以原图为准。",
        "- **表格内容不要依赖文本层**：扫描表格 OCR 后行列关系丢失、数字易错。",
        "  5210.4 的质量验收表、记录表请直接看原图。",
    ]
    wm = ["%s 第 %s 页" % (v["code"], "、".join(str(x) for x in v["watermark_pages"]))
          for v in stats.values() if v["watermark_pages"]]
    if wm:
        L.append("- **部分页面在源 PDF 中即带橙色印章/水印，会遮挡少量表格内容**：%s。" % "；".join(wm))
    emp = ["%s 第 %s 页" % (v["code"], "、".join(str(x) for x in v["empty_pages"]))
           for v in stats.values() if v["empty_pages"]]
    if emp:
        L.append("- 少数页面无可用文本层（多为整页图或空白页），请直接看原图：%s。" % "；".join(emp))
    rev = ["%s 第 %s 页" % (v["code"], "、".join(str(x) for x in v["review_pages"]))
           for v in stats.values() if v.get("review_pages")]
    if rev:
        L.append("- **建议人工复核的页**（自动选层时语言模型倾向另一层，已按位置证据保留了较优的一层，"
                 "但该页文本质量存疑，引用时请核对原图）：%s。" % "；".join(rev))
    L += [
        "- PDF 页码 ≠ 原书印刷页码。印刷页码见 `00_总览/页面清单.csv`。",
        "",
        "## 重新生成",
        "",
        "依赖 PyMuPDF、Pillow、NumPy。",
        "",
        "```powershell",
        "$py = \"C:\\Users\\田浩\\.dsh\\dsh-runtimes\\dsh-primary-runtime\\dependencies\\python\\python.exe\"",
        "& $py -X utf8 \"工具\\build_kb.py\"",
        "```",
        "",
    ]
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
