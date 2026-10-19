# -*- coding: utf-8 -*-
"""favs — B 站收藏夹：① 同步到本地清单（打开第 N 条要用）② 打开第 N 条 / 看第 N 条

为什么单独一个模块：掩码表只能匹配【固定说法】，「打开第 5 条」里的数字是变量，
所以那类句子走【模式匹配】（见 intents.PATTERNS）→ 动作 arg = "文件夹|序号"。
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "fav_cache"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
UID = "389542160"
FOLDERS = [("默认收藏夹", 297290160), ("课程", 1386162960),
           ("哄up收藏夹", 844167860), ("明日方舟", 817457860)]
ALIAS = {"": "默认收藏夹", "默认": "默认收藏夹", "默认收藏夹": "默认收藏夹",
         "课程": "课程", "哄up": "哄up收藏夹", "哄up收藏夹": "哄up收藏夹",
         "明日方舟": "明日方舟"}
HDR = {"User-Agent": UA,
       "Referer": f"https://space.bilibili.com/{UID}/favlist",
       "Accept": "application/json, text/plain, */*"}


def _log(msg, tag="SYS", level=None):
    try:
        from dlog import log
        log(msg, tag, level=level)
    except Exception:
        print(msg, flush=True)


def _cache_file(name):
    safe = "".join(c for c in name if c.isalnum() or c in "_-")
    return CACHE_DIR / f"{safe}.json"


def load_cache(name):
    p = _cache_file(name)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


_BUV = {"v": None, "ts": 0.0}


def _buvid():
    """匿名设备标识 buvid3/buvid4 —— 批量翻页被 412 风控时必需（一小时缓存）"""
    if _BUV["v"] and time.time() - _BUV["ts"] < 3600:
        return _BUV["v"]
    try:
        # ⚠️ 这里【不能】走 _api（_api 里会调 _buvid → 无限递归），直接发一次
        req = urllib.request.Request(
            "https://api.bilibili.com/x/frontend/finger/spi", headers=HDR)
        with urllib.request.urlopen(req, timeout=15) as r:
            j = json.loads(r.read().decode("utf-8", "replace"))
        d = j.get("data") or {}
        b3, b4 = d.get("b_3"), d.get("b_4")
        if b3:
            _BUV["v"] = (b3, b4 or "")
            _BUV["ts"] = time.time()
            _log(f"   🍪 已获取匿名 buvid3（{str(b3)[:14]}…）")
    except Exception as e:
        _log(f"   ⚠️ 取 buvid 失败: {type(e).__name__}: {e}", "WARN")
    return _BUV["v"]


def page_of(n, ps=20):
    return max(1, (int(n) - 1) // ps + 1)


def fetch_page(name, pn, ps=20, page_delay=0.0, retry=3):
    """只取某一页（1 次请求）—— 「打开第 N 条」用这个，不必整份同步（避开风控）

    结果按页缓存到 fav_cache/<folder>_p<N>.json（1 小时内复用）。
    """
    import json as _json
    media_id = dict(FOLDERS).get(name)
    if not media_id:
        return None
    cache = CACHE_DIR / f"{_cache_file(name).stem}_p{int(pn)}.json"
    try:
        st = cache.stat()
        if time.time() - st.st_mtime < 3600:
            return _json.loads(cache.read_text(encoding="utf-8"))
    except Exception:
        pass
    r = _api("/x/v3/fav/resource/list", media_id=media_id, pn=int(pn), ps=int(ps),
             order="mtime", type=0, platform="web", _retry=retry)
    if r.get("code") != 0:
        return {"code": r.get("code"), "message": r.get("message"), "items": []}
    d = r.get("data") or {}
    info = d.get("info") or {}
    items = []
    for k, it in enumerate(d.get("medias") or []):
        idx = (int(pn) - 1) * int(ps) + k + 1
        if not it:
            items.append({"i": idx, "title": "（失效/已删除）", "url": ""})
            continue
        dur = int(it.get("duration") or 0)
        items.append({
            "i": idx,
            "title": it.get("title") or "",
            "bvid": it.get("bvid") or "",
            "url": f"https://www.bilibili.com/video/{it.get('bvid')}",
            "up": ((it.get("upper") or {}).get("name")) or "",
            "dur": f"{dur // 60}:{dur % 60:02d}",
            "intro": (it.get("intro") or "")[:120],
        })
    out = {"folder": name, "media_id": media_id, "page": int(pn), "ps": int(ps),
           "declared": int(info.get("media_count") or 0), "count": len(items),
           "items": items, "ts": time.time()}
    if page_delay:
        time.sleep(page_delay)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(_json.dumps(out, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return out


def get_item(name, n, ps=20):
    """第 N 条 → {"title","bvid","url",...}；取不到返回 (None, 原因)"""
    try:
        n = int(n)
    except Exception:
        return None, f"没听懂序号 {n!r}"
    pn = page_of(n, ps)
    pg = fetch_page(name, pn, ps=ps)
    if not pg:
        return None, f"收藏夹「{name}」不存在"
    if pg.get("code", 0) != 0:
        return None, (f"拿第 {n} 条失败：{pg.get('code')} {pg.get('message')}"
                      f"（风控/网络；稍后再试或说「刷新收藏夹」）")
    for it in pg.get("items") or []:
        if int(it.get("i") or 0) == n:
            return it, None
    declared = pg.get("declared") or 0
    if declared and n > declared:
        return None, f"「{name}」只有 {declared} 条，第 {n} 条不存在"
    return None, f"「{name}」第 {n} 条这一页没取到（页 {pn}）"


def _api(path, _retry=3, **_kw):
    """带 buvid3 + 412 退避重试"""
    import urllib.error
    u = f"https://api.bilibili.com{path}?{urllib.parse.urlencode(_kw)}"
    hdr = dict(HDR)
    bv = _buvid()
    if bv:
        hdr["Cookie"] = f"buvid3={bv[0]}; buvid4={bv[1]}"
    for attempt in range(_retry + 1):
        try:
            req = urllib.request.Request(u, headers=hdr)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code == 412:
                wait = 8 * (attempt + 1)
                _log(f"   ⚠️ 412 风控，等 {wait}s 重试（{attempt + 1}/{_retry}）…", "WARN")
                time.sleep(wait)
                continue
            return {"code": -e.code, "message": f"HTTP {e.code}"}
        except Exception as e:
            if attempt < _retry:
                time.sleep(3 * (attempt + 1))
                continue
            return {"code": -1, "message": f"{type(e).__name__}: {e}"}
    return {"code": -412, "message": "412 风控（多次重试仍失败）"}


def sync_one(name, media_id=None, page_delay=1.2, max_pages=500):
    """把一个收藏夹整份拉到本地（含标题/bvid/链接/UP/时长）

    节奏：每页 1.2s（B 站对匿名批量翻页有风控，快了直接 412）；每 40 页落一次盘，
    所以即使中途被风控挡住，已拿到的部分也能用（partial=true）。
    """
    if media_id is None:
        for n, m in FOLDERS:
            if n == name:
                media_id = m
                break
    if not media_id:
        return None
    items, pn, total, partial = [], 1, None, False
    while pn <= max_pages:
        r = _api("/x/v3/fav/resource/list", media_id=media_id, pn=pn, ps=20,
                 order="mtime", type=0, platform="web")
        if r.get("code") != 0:
            _log(f"   ⚠️ 收藏夹「{name}」第 {pn} 页失败: {r.get('code')} {r.get('message')}"
                 f"（已拿到 {len(items)} 条，先存下）", "WARN")
            partial = True
            break
        d = r.get("data") or {}
        info = d.get("info") or {}
        if total is None:
            total = int(info.get("media_count") or 0)
        med = d.get("medias") or []
        if not med:
            break
        for it in med:
            if not it:
                items.append({"i": len(items) + 1, "title": "（失效/已删除）", "url": ""})
                continue
            dur = int(it.get("duration") or 0)
            items.append({
                "i": len(items) + 1,
                "title": it.get("title") or "",
                "bvid": it.get("bvid") or "",
                "url": f"https://www.bilibili.com/video/{it.get('bvid')}",
                "up": ((it.get("upper") or {}).get("name")) or "",
                "dur": f"{dur // 60}:{dur % 60:02d}",
                "intro": (it.get("intro") or "")[:120],
            })
        if pn % 40 == 0 or len(items) >= (total or 0):
            _save(name, media_id, items, total, partial=True)
            _log(f"   📚 「{name}」已拿 {len(items)}/{total} 条…", level=None)
        if len(items) >= (total or 0):
            break
        pn += 1
        if page_delay:
            time.sleep(page_delay)
    return _save(name, media_id, items, total, partial=partial)


def _save(name, media_id, items, total, partial=False):
    data = {"folder": name, "media_id": media_id, "count": len(items),
            "declared": total, "partial": bool(partial), "ts": time.time(),
            "url": f"https://space.bilibili.com/{UID}/favlist?fid={media_id}&ftype=create",
            "items": items}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_file(name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    md = [f"# B站收藏夹：{name}（{len(items)}/{total} 条，"
          f"{time.strftime('%Y-%m-%d %H:%M')}{'，部分' if partial else ''}）", ""]
    md += [f"{it['i']}. [{it['title']}]({it['url']}) ｜ {it['dur']} ｜ {it['up']}"
           for it in items]
    (_cache_file(name).with_suffix(".md")).write_text("\n".join(md), encoding="utf-8")
    return data


def sync_async(names=None, page_delay=0.35):
    """后台整份同步（不阻塞语音主循环）"""
    def _work():
        t0 = time.time()
        for n in (names or [n for n, _ in FOLDERS]):
            try:
                d = sync_one(n, page_delay=page_delay)
                if d:
                    _log(f"   📚 收藏夹「{n}」已同步：{d['count']} 条"
                         f"（声明 {d['declared']}）→ {_cache_file(n)}", level="info")
            except Exception as e:
                _log(f"   ⚠️ 收藏夹「{n}」同步失败: {type(e).__name__}: {e}", "WARN")
        _log(f"   📚 收藏夹同步完成（{time.time() - t0:.0f}s）", level="info")

    t = threading.Thread(target=_work, daemon=True, name="oc-favs")
    t.start()
    return t


# ── 意图动作（dispatch 签名 (arg, cfg, dry=False) ）────────────────
def _resolve(arg):
    """arg = '文件夹|序号' 或 '文件夹'（打开页面）或 '5'（默认夹第 5 条）"""
    a = str(arg or "").strip()
    folder, n = "", None
    if "|" in a:
        folder, _, ns = a.partition("|")
        n = ns.strip() or None
    else:
        n = a or None
    return ALIAS.get(folder.strip(), "默认收藏夹"), n


def act_fav_open(arg, cfg, dry=False):
    name, ns = _resolve(arg)
    if not ns:                       # 只给文件夹 → 打开收藏夹页面
        url = f"https://space.bilibili.com/{UID}/favlist?fid={dict(FOLDERS)[name]}&ftype=create"
        if dry:
            return True, f"[dry] 打开收藏夹页面：{url}", {"url": url}
        os.startfile(url)
        return True, f"已打开收藏夹「{name}」", {"url": url}
    n = cn2int(ns)
    if not n:
        return False, f"没听懂序号「{ns}」", None
    it, err = get_item(name, n)
    if it is None:
        return False, err, None
    if not it.get("url"):
        return False, f"「{name}」第 {n} 条是失效视频（无链接）", None
    if dry:
        return True, f"[dry] 打开第 {n} 条：{it['title']} → {it['url']}", it
    os.startfile(it["url"])
    return True, (f"已打开「{name}」第 {n} 条：{it['title']}"
                  f"（{it.get('dur')} ｜ {it.get('up')}）"), it


def act_fav_show(arg, cfg, dry=False):
    name, ns = _resolve(arg)
    if not ns:
        pg = fetch_page(name, 1)
        d = pg or {}
        return True, (f"「{name}」共 {d.get('declared') or '?'} 条"
                      f"（本地只按需取页，不整份同步）"), d
    n = cn2int(ns)
    if not n:
        return False, f"没听懂序号「{ns}」", None
    it, err = get_item(name, n)
    if it is None:
        return False, err, None
    return True, (f"「{name}」第 {n} 条：{it['title']}（{it.get('dur')} ｜ "
                  f"{it.get('up')}）｜ {it.get('url')}"), it


def act_fav_sync(arg, cfg, dry=False):
    names = None
    if arg:
        nm = ALIAS.get(str(arg).strip())
        names = [nm] if nm else None
    if dry:
        return True, f"[dry] 后台同步收藏夹 {names or '（全部）'}", None
    sync_async(names)
    return True, (f"已在后台同步收藏夹{'（' + '、'.join(names) + '）' if names else '（全部 4 个）'}"
                  f"；完成后会打一行日志（说「收藏夹」可看进度/条数）"), None


def cn2int(s):
    """中文/阿拉伯数字 → int（五→5、十二→12、一百零五→105）"""
    s = str(s or "").strip()
    if s.isdigit():
        return int(s)
    units = {"十": 10, "百": 100, "千": 1000}
    digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    total, num = 0, 0
    for ch in s:
        if ch in digits:
            num = digits[ch]
        elif ch in units:
            total += (num or 1) * units[ch]
            num = 0
        else:
            return None
    return total + num


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "sync":
        for n, _ in FOLDERS:
            d = sync_one(n)
            print(f"{n}: {d['count'] if d else '失败'} 条")
    else:
        for n, _ in FOLDERS:
            d = load_cache(n)
            print(f"{n}: {'未同步' if d is None else str(d['count']) + ' 条'}")
        print("中文数字:", [(s, cn2int(s)) for s in ("五", "十二", "二十", "二十三", "一百零五", "30")])
