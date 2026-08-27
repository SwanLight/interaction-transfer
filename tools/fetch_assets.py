#!/usr/bin/env python3
"""按前缀从 Omniverse S3 拉取 Isaac 5.1 资产到 /mnt/isaacsim_assets，保持目录结构。"""
import os, sys, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
DEST = "/mnt/isaacsim_assets"

PREFIXES = [
    "Assets/Isaac/5.1/Isaac/Robots/WonikRobotics/AllegroHand/",
    "Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/",
    "Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/",
    "Assets/Isaac/5.1/Isaac/Props/Sektion_Cabinet/",
    "Assets/Isaac/5.1/Isaac/Props/Mounts/SeattleLabTable/",
    "Assets/Isaac/5.1/Isaac/Props/Blocks/",
]

def list_keys(prefix):
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token: q["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}", timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall(f"{NS}Contents"):
            k = c.find(f"{NS}Key").text
            s = int(c.find(f"{NS}Size").text)
            if not k.endswith("/"):
                keys.append((k, s))
        if root.findtext(f"{NS}IsTruncated") != "true": break
        token = root.findtext(f"{NS}NextContinuationToken")
    return keys

def fetch(item):
    key, size = item
    dst = os.path.join(DEST, key)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return ("skip", key, size)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    try:
        urllib.request.urlretrieve(url, dst)
        return ("ok", key, size)
    except Exception as e:
        return ("FAIL", key, str(e))

all_keys = []
for p in PREFIXES:
    ks = list_keys(p)
    tot = sum(s for _, s in ks)
    print(f"{len(ks):5d} files  {tot/1048576:9.1f} MB   {p}", flush=True)
    all_keys += ks

print(f"\n合计 {len(all_keys)} 文件 / {sum(s for _,s in all_keys)/1048576:.1f} MB，开始下载...\n", flush=True)

ok = skip = fail = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    for status, key, info in ex.map(fetch, all_keys):
        if status == "ok": ok += 1
        elif status == "skip": skip += 1
        else:
            fail += 1
            print(f"FAIL {key}: {info}", flush=True)

print(f"\n完成: 新下载 {ok} / 已存在 {skip} / 失败 {fail}")
