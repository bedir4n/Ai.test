#!/usr/bin/env python3
"""Gerçek Kiso vadisi yükseklik referansını indir (AWS Open Data "Terrain Tiles", terrarium kodlaması).

Narai-juku ve Tsumago-juku çevresi, zoom 15 (~3.9 m/px). Çıktı: kiso_reference.npz
  narai, tsumago : float32 yükseklik (m), res_m: piksel boyu
heightmap.py yamaç kabartma dokusunu ve eğim kalibrasyonunu bu veriden alır.
"""
import io, math, os, urllib.request
import numpy as np
from PIL import Image

URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/15/{x}/{y}.png"
SITES = {"narai": (35.9665, 137.8135), "tsumago": (35.5770, 137.5960)}


def tile(lat, lon, z=15):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return int(x), int(y)


def get(x, y):
    a = np.array(Image.open(io.BytesIO(urllib.request.urlopen(URL.format(x=x, y=y), timeout=30).read())).convert("RGB"), float)
    return a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768


out = {}
for name, (lat, lon) in SITES.items():
    tx, ty = tile(lat, lon)
    out[name] = np.vstack([np.hstack([get(tx + dx, ty + dy) for dx in (-1, 0, 1)]) for dy in (-1, 0, 1)]).astype(np.float32)
    print(name, out[name].shape, out[name].min(), out[name].max())
out["res_m"] = np.float32(156543.03 / 2 ** 15 * math.cos(math.radians(35.8)))
np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kiso_reference.npz"), **out)
