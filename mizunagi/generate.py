#!/usr/bin/env python3
"""Mizunagi-cho kasaba planı üreticisi.

Ölçek: gerçek metre. Harita 440 x 352 m, çıktı 2.5 px/m (1100 x 880 px).
Sıra: arazi bölgeleri -> yollar -> özel yapılar -> sokak boyunca parseller
-> sokak donatıları (lamba, direk) -> tarla/orman dolgusu -> QA -> çizim.

Çalıştır:  python3 generate.py   (çıktı: out/ klasörü)
"""
import json, math, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 440.0, 352.0
PX = 2.5                       # px / m
RES = 2.0                      # çarpışma ızgarası: hücre / m
GW, GH = int(W * RES), int(H * RES)
rnd = random.Random(1991)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- yardımcılar
def cell(x, y):
    return int(x * RES), int(y * RES)

def stamp_line(mask, pts, width, val=True):
    """Polyline'ı ızgaraya basar (dolu daire fırçası)."""
    r = width / 2
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) * RES))
        for i in range(n + 1):
            t = i / n
            stamp_disc(mask, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, r, val)

def stamp_disc(mask, x, y, r, val=True):
    cx, cy = x * RES, y * RES
    rr = r * RES
    x0, x1 = max(0, int(cx - rr) - 1), min(GW, int(cx + rr) + 2)
    y0, y1 = max(0, int(cy - rr) - 1), min(GH, int(cy + rr) + 2)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask[y0:y1, x0:x1][(xx + .5 - cx) ** 2 + (yy + .5 - cy) ** 2 <= rr * rr] = val

def rect_pts(cx, cy, w, d, ang, step=0.5):
    """Dönük dikdörtgenin içindeki örnek noktaları (metre)."""
    us = np.arange(-w / 2, w / 2 + 1e-9, step)
    vs = np.arange(-d / 2, d / 2 + 1e-9, step)
    u, v = np.meshgrid(us, vs)
    c, s = math.cos(ang), math.sin(ang)
    return cx + u * c - v * s, cy + u * s + v * c

def rect_corners(cx, cy, w, d, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [(cx + u * c - v * s, cy + u * s + v * c)
            for u, v in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2))]

def hit(mask, xs, ys):
    ix = np.clip((xs * RES).astype(int), 0, GW - 1)
    iy = np.clip((ys * RES).astype(int), 0, GH - 1)
    out = (xs < 0) | (xs >= W) | (ys < 0) | (ys >= H)
    return bool(np.any(mask[iy, ix]) or np.any(out))

def inside(mask, x, y):
    return bool(mask[min(GH - 1, max(0, int(y * RES))), min(GW - 1, max(0, int(x * RES)))])

def polylen(pts):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))

def walk(pts, step):
    """Polyline boyunca (x, y, açı) üretir."""
    out, carry = [], 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        if L == 0:
            continue
        a = math.atan2(y1 - y0, x1 - x0)
        d = carry
        while d <= L:
            out.append((x0 + (x1 - x0) * d / L, y0 + (y1 - y0) * d / L, a))
            d += step
        carry = d - L
    return out

# ----------------------------------------------------------------- ızgaralar
road = np.zeros((GH, GW), bool)      # yol + yol payı
solid = np.zeros((GH, GW), bool)     # nehir, bina, özel alan (bina koyma)
river_m = np.zeros((GH, GW), bool)
field = np.zeros((GH, GW), bool)     # tarla bölgesi
forest_z = np.zeros((GH, GW), bool)  # orman bölgesi (ev yok)
lots = np.zeros((GH, GW), bool)      # otopark/avlu vb. (ev yok, yol değil)

# ------------------------------------------------------------------- nehir
RIVER = [(70, -5), (80, 40), (90, 90), (84, 140), (72, 185), (74, 225), (82, 270), (92, 310), (90, 357)]
RIVER_W = 11
stamp_line(river_m, RIVER, RIVER_W)
stamp_line(solid, RIVER, RIVER_W + 8)       # set payı dahil: ev yok
LEVEE_L = [(x - 8, y) for x, y in RIVER]
LEVEE_R = [(x + 8, y) for x, y in RIVER]

# ------------------------------------------------------------------- yollar
# (ad, tür, genişlik m, noktalar)
ROADS = [
    ("Chuo-dori", "road", 7, [(225, 318), (225, 215)]),
    ("Sando (tapınak yolu)", "road", 5, [(225, 215), (225, 150), (225, 112)]),
    ("Old Kaido", "road", 6.5, [(0, 218), (62, 218), (98, 218), (225, 218)]),
    ("East Road", "road", 6.5, [(225, 218), (290, 230), (350, 233), (405, 238), (418, 262)]),
    ("Mountain Road", "road", 5.5, [(228, 210), (285, 168), (345, 138), (400, 112), (440, 100)]),
    ("Shotengai", "road", 6, [(160, 268), (225, 268), (300, 268)]),
    ("Station Front", "road", 6, [(103, 318), (225, 318), (330, 318)]),
    ("River Road", "road", 6, [(100, 218), (102, 270), (106, 318)]),
    ("West Bank Road", "alley", 4, [(40, 150), (40, 218), (42, 260), (45, 330)]),
    ("Furumachi A", "alley", 3.2, [(122, 170), (225, 170)]),
    ("Furumachi B", "alley", 3.2, [(122, 196), (225, 196)]),
    ("Furumachi 1", "alley", 3.2, [(122, 170), (122, 218)]),
    ("Furumachi 2", "alley", 3.2, [(165, 170), (165, 218)]),
    ("Furumachi 3", "alley", 3.2, [(195, 150), (195, 196)]),
    ("Roji W1", "alley", 3.2, [(135, 218), (135, 318)]),
    ("Roji W2", "alley", 3.2, [(170, 218), (170, 318)]),
    ("Roji W3", "alley", 3.2, [(198, 268), (198, 318)]),
    ("Roji W-ara", "alley", 3.2, [(110, 245), (160, 245), (225, 245)]),
    ("Roji E1", "alley", 3.2, [(250, 238), (250, 268)]),
    ("Roji E2", "alley", 3.2, [(275, 234), (275, 268)]),
    ("Roji E3", "alley", 3.2, [(300, 232), (300, 318)]),
    ("Roji E-ara", "alley", 3.2, [(225, 248), (300, 248), (330, 248)]),
    ("Clinic Road", "road", 5, [(330, 236), (330, 280), (330, 318)]),
    ("School Road", "road", 5, [(362, 233), (362, 218)]),
    ("Farm Lane S1", "dirt", 3.6, [(330, 292), (380, 292), (432, 292)]),
    ("Farm Lane S2", "dirt", 3.6, [(380, 245), (380, 292), (380, 330)]),
    ("Farm Lane S3", "dirt", 3.6, [(110, 318), (110, 350)]),
    ("Farm Lane W", "dirt", 3.6, [(8, 175), (8, 345)]),
    ("Forest Link", "dirt", 3.6, [(345, 138), (372, 95), (395, 62)]),
    ("Kamiyama Rd Ext", "road", 6.5, [(0, 218), (-1, 218)]),
    ("Bike Trail", "trail", 3.2, [(232, 205), (258, 180), (300, 160), (345, 148)]),
]
road_edges = {}
for name, kind, wd, pts in ROADS:
    stamp_line(road, pts, wd + (1.0 if kind != "trail" else 0))
BRIDGE = [(62, 218), (98, 218)]  # Old Kaido köprüsü (nehri yol sayma)
# köprü: nehir üstünde yol var, ev yok -> nehir zaten solid

# demiryolu
RAIL_Y = 338
stamp_line(road, [(0, RAIL_Y), (440, RAIL_Y)], 6)
RAIL_BRIDGE = (80, 100)

# ------------------------------------------------------------ arazi bölgeleri
def rect_mask(mask, x0, y0, x1, y1, val=True):
    mask[int(y0 * RES):int(y1 * RES), int(x0 * RES):int(x1 * RES)] = val

# orman: kuzey şerit, kuzeydoğu yamaç, batı yamaç
rect_mask(forest_z, 0, 0, 440, 120)
rect_mask(forest_z, 300, 0, 440, 170)
rect_mask(forest_z, 100, 0, 300, 150)
rect_mask(forest_z, 0, 0, 62, 150)
rect_mask(forest_z, 92, 0, 440, 124)
# tarlalar: güneydoğu, doğu-orta, nehir batı yakası, güney-orta boşluğu
FIELDS = [(362, 252, 438, 330), (400, 125, 438, 190),
          (3, 148, 36, 335), (48, 150, 62, 335), (112, 322, 215, 336), (228, 322, 336, 336)]
FIELDS = [f for f in FIELDS if f[3] - f[1] > 6]
# tarla bölgesi: sadece yol ve yapı olmayan yerlerde çizilir, ev konmaz
for f in FIELDS:
    rect_mask(field, *f)
field &= ~road
# (112,322)-(336) şeridi demiryolu ile istasyon yolu arasındaki dar bant: yerine çayır
for f in [(112, 322, 215, 336), (228, 322, 336, 336)]:
    rect_mask(field, *f, val=False)

# ------------------------------------------------------- özel yapılar listesi
buildings = []   # {id,type,x,y,w,d,ang,label,floors}
areas = []       # {type,poly,label}  otopark, avlu, park...
labels = []      # (x,y,text)

SPECIAL = [True]

def free_rect(cx, cy, w, d, ang, margin=1.0, ignore_forest=False, allow_field=False):
    xs, ys = rect_pts(cx, cy, w + 2 * margin, d + 2 * margin, ang)
    if hit(road, xs, ys) or hit(solid, xs, ys) or (not SPECIAL[0] and hit(lots, xs, ys)):
        return False
    if not allow_field and hit(field, xs, ys):
        return False
    if not ignore_forest and hit(forest_z, xs, ys):
        return False
    return True

def place(typ, cx, cy, w, d, ang=0.0, label=None, floors=1, margin=1.0, nudge=True,
          ignore_forest=False, allow_field=False):
    tries = [(0, 0)]
    if nudge:
        for r in (3, 6, 9, 12):
            for k in range(8):
                a = k * math.pi / 4
                tries.append((r * math.cos(a), r * math.sin(a)))
    for dx, dy in tries:
        if free_rect(cx + dx, cy + dy, w, d, ang, margin, ignore_forest, allow_field):
            b = dict(id=len(buildings), type=typ, x=round(cx + dx, 2), y=round(cy + dy, 2),
                     w=w, d=d, ang=round(ang, 4), label=label, floors=floors)
            buildings.append(b)
            xs, ys = rect_pts(b["x"], b["y"], w + .2, d + .2, ang)
            ix = np.clip((xs * RES).astype(int), 0, GW - 1)
            iy = np.clip((ys * RES).astype(int), 0, GH - 1)
            solid[iy, ix] = True
            if label:
                labels.append((b["x"], b["y"] - d / 2 - 2.5, label))
            return b
    if label:
        print("UYARI: yerleşmedi:", typ, label)
    return None

def area(typ, x0, y0, x1, y1, label=None, block=True):
    areas.append(dict(type=typ, rect=[x0, y0, x1, y1], label=label))
    if block:
        rect_mask(lots, x0, y0, x1, y1)

# --- tapınak terası (orman içinde, doğal teras)
SH = (225, 62)
area("shrine_court", 211, 40, 242, 90, "Shrine", block=True)
place("shrine_hall", 225, 58, 11, 9, 0, "Haiden", 1, ignore_forest=True, nudge=False)
place("honden", 225, 46, 6, 6, 0, None, 1, ignore_forest=True, nudge=False)
place("shrine_office", 235, 80, 8, 6, 0, "Shrine office", 1, ignore_forest=True, nudge=False)
place("torii", 225, 108, 6, 1.2, 0, "Torii", 1, ignore_forest=True, nudge=False, margin=0)
area("steps", 221.5, 100, 228.5, 130, "108 steps  +16 m", block=False)
# --- tapınak + mezarlık ayrı teras
area("temple_court", 126, 52, 164, 80, "Temple", block=True)
place("temple", 145, 64, 16, 11, 0, "Temple", 1, ignore_forest=True, nudge=False)
place("priest_house", 180, 66, 11, 8, 0, "Priest's house", 1, ignore_forest=True, nudge=False)
area("cemetery", 128, 92, 164, 122, "Cemetery", block=True)
# --- istasyon bölgesi
area("station_plaza", 203, 322, 247, 336, None, block=False)
place("station", 225, 329.5, 22, 7, 0, "Mizunagi Station", 1, margin=0.5, nudge=False)
place("post_office", 184, 302, 20, 12, 0, "Post Office", 2, nudge=False)
place("bike_garage", 192, 328, 10, 6, 0, "Bike garage", 1, nudge=False)
place("ja_coop", 240, 305, 20, 11, 0, "JA co-op (rice)", 1, nudge=False)
place("freight_shed", 300, 328, 20, 7, 0, "Freight shed", 1, nudge=False)
area("parking", 231, 276, 250, 292, "P  station", block=True)
# --- ana cadde ticari
place("konbini", 238, 232, 11, 9, 0, "Konbini", 1, nudge=False)
area("parking", 242, 237, 268, 247, "P", block=True)
place("ramen", 212, 232, 9, 8, 0, "Ramen-ya", 1)
place("police_box", 212, 205, 6, 5, 0, "Police box", 1)
place("town_hall", 195, 207, 20, 11, 0, "Town Hall", 2, nudge=False)
place("market", 262, 257.3, 15, 10, 0, "Market", 1)
place("supermarket", 277, 290, 30, 16, 0, "Supermarket", 1, nudge=False)
area("parking", 254, 300, 296, 311, "P  supermarket", block=True)
place("barber", 205, 258, 8, 7, 0, "Barber", 1)
place("bicycle_shop", 186, 258, 9, 8, 0, "Bicycle shop", 1)
place("bakery", 245, 258, 8, 7, 0, "Bakery", 1)
place("sento", 125, 255, 14, 11, 0, "Sento (bath)", 2)
area("park", 108, 275, 128, 295, "Playground", block=True)
place("fire_station", 314, 258, 16, 12, 0, "Fire station", 2)
place("clinic", 345, 263, 14, 11, 0, "Clinic", 2)
area("parking", 335, 277, 358, 289, "P", block=True)
# --- okul
area("schoolyard", 346, 172, 398, 197, "Schoolyard", block=True)
place("school", 372, 204, 54, 12, 0, "Mizunagi Primary School", 2, nudge=False, margin=0.8)
place("gym", 330, 190, 18, 14, 0, "gym", 1)
place("community_hall", 408, 222, 18, 12, 0, "Community hall", 1)
place("water_tower", 392, 122, 6, 6, 0, "Water tower", 1, ignore_forest=True)
place("forestry_hut", 328, 128, 10, 7, 0.4, "Forestry hut", 1, ignore_forest=True)
place("lookout", 388, 98, 6, 6, 0, "Lookout rock", 1, ignore_forest=True)
place("radio_mast", 372, 70, 5, 5, 0, "Radio mast", 1, ignore_forest=True)
place("rest_hut", 410, 62, 7, 6, 0, "Old postmen's rest hut", 1, ignore_forest=True)
labels.append((425, 28, "Old tunnel (sealed)"))
SPECIAL[0] = True
# --- dağ evleri (Mountain Road yanında, orman kenarı)
for i, (mx, my) in enumerate([(338, 150), (352, 145), (320, 160), (300, 175)]):
    place("house", mx, my, 8, 7, 0.5, "Mountain houses" if i == 0 else None, 1,
          ignore_forest=True)

# ------------------------------------------- sokak boyunca parsel (şerit imar)
HOUSE_KINDS = [("house", .62), ("machiya", .12), ("apartment", .08), ("shop", .08), ("warehouse", .05),
               ("abandoned", .05)]

def pick_kind(zone):
    r, acc = rnd.random(), 0
    kinds = HOUSE_KINDS
    if zone == "furumachi":
        kinds = [("machiya", .45), ("house", .38), ("abandoned", .12), ("warehouse", .05)]
    if zone == "shop":
        kinds = [("shop", .6), ("machiya", .2), ("house", .1), ("apartment", .1)]
    for k, p in kinds:
        acc += p
        if r <= acc:
            return k
    return "house"

SIZE = {"house": (7, 9, 7, 10, 1), "machiya": (5, 7, 10, 14, 2), "apartment": (10, 14, 10, 14, 3),
        "shop": (6, 9, 8, 11, 1), "warehouse": (9, 12, 8, 12, 1), "abandoned": (6, 8, 7, 9, 1)}

SHOP_ROADS = {"Shotengai", "Chuo-dori", "East Road", "Station Front"}
SKIP = {"Sando (tapınak yolu)", "Mountain Road", "Forest Link", "Bike Trail",
        "Kamiyama Rd Ext", "Farm Lane W", "Farm Lane S1", "Farm Lane S2", "Farm Lane S3",
        "School Road"}

SPECIAL[0] = False
for name, kind, wd, pts in sorted(ROADS, key=lambda r: -r[2]):
    if name in SKIP:
        continue
    zone = "shop" if name in SHOP_ROADS else ("furumachi" if name.startswith("Furumachi") else "res")
    gap = 0.6
    for side in (-1, 1):
        for x, y, a in walk(pts, 1.0):
            pass
        # kenar boyunca ilerle: her adımda ön cephe uzunluğu kadar atla
        samples = walk(pts, 1.0)
        i = 0
        while i < len(samples):
            x, y, a = samples[i]
            k = pick_kind(zone)
            fw0, fw1, d0, d1, fl = SIZE[k]
            fw = rnd.uniform(fw0, fw1)
            d = rnd.uniform(d0, d1)
            setback = (1.6 if kind == "alley" else 2.4 if zone != "shop" else 1.4)
            off = wd / 2 + 0.5 + setback + d / 2
            nx, ny = -math.sin(a) * side, math.cos(a) * side
            adv = fw / 2
            j = min(len(samples) - 1, i + int(adv))
            sx, sy, sa = samples[j]
            cx, cy = sx + nx * off, sy + ny * off
            # bina uzun ekseni yola paralel; hafif sapma yok (plan düzeni)
            ang = sa
            b = place(k, cx, cy, fw, d, ang, None, fl, margin=gap, nudge=False)
            i = j + int(fw / 2 + 2 * gap + 0.5) + (0 if b else 3)
            if b and zone != "shop" and rnd.random() < .12:
                i += rnd.randint(4, 9)   # arada boşluk / bahçe

# --- ikinci sıra: sokak arkasındaki dolgu (alley'lerden erişilen evler)
for _ in range(9000):
    x, y = rnd.uniform(100, 335), rnd.uniform(150, 316)
    k = pick_kind("furumachi" if y < 218 and x < 230 else "res")
    fw0, fw1, d0, d1, fl = SIZE[k]
    ang = 0.0 if rnd.random() < .5 else math.pi / 2
    # hedef: en yakın sokağa cephe; ızgara yönü yeterli
    place(k, x, y, rnd.uniform(fw0, fw1), rnd.uniform(d0, d1), ang, None, fl, margin=1.3, nudge=False)

# --- ev bahçesi: batı yakada tek tük ev (çiftlik evleri)
for _ in range(60):
    x, y = rnd.uniform(5, 60), rnd.uniform(150, 330)
    place("farmhouse", x, y, rnd.uniform(9, 12), rnd.uniform(8, 10), 0, None, 1, margin=1.2,
          nudge=False, allow_field=True)
for _ in range(80):
    x, y = rnd.uniform(336, 436), rnd.uniform(255, 332)
    if rnd.random() < .12:
        place("farmhouse", x, y, rnd.uniform(9, 12), rnd.uniform(8, 10), 0, None, 1, margin=1.5,
              nudge=False, allow_field=True)

# ------------------------------------------------------------ sokak donatıları
lamps, poles, vend, bins = [], [], [], []
for name, kind, wd, pts in ROADS:
    if kind == "trail" or name in ("Kamiyama Rd Ext",):
        continue
    step = 22 if kind in ("road",) else 30
    if kind == "dirt":
        step = 45
    for n, (x, y, a) in enumerate(walk(pts, step)):
        side = 1 if n % 2 == 0 else -1
        off = wd / 2 + 0.7
        px_, py_ = x - math.sin(a) * side * off, y + math.cos(a) * side * off
        if hit(solid, np.array([px_]), np.array([py_])) and not hit(river_m, np.array([px_]), np.array([py_])):
            pass
        if hit(river_m, np.array([px_]), np.array([py_])):
            continue
        # bina içine düşmesin
        if any(abs(px_ - b["x"]) < b["w"] / 2 + .3 and abs(py_ - b["y"]) < b["d"] / 2 + .3
               for b in buildings if abs(b["x"] - px_) < 30 and abs(b["y"] - py_) < 30 and b["ang"] % (math.pi / 2) < 1e-3):
            continue
        (lamps if kind == "road" else poles).append((round(px_, 1), round(py_, 1)))
# otomatlar / çöp kutuları ticari bölgede
for b in buildings:
    if b["type"] in ("konbini", "shop", "supermarket", "station") and rnd.random() < .85:
        vend.append((round(b["x"] + b["w"] / 2 + 1.0, 1), round(b["y"] + b["d"] / 2 - 1.2, 1)))

# ------------------------------------------------------------ QA
def qa():
    rep = {"buildings": len(buildings)}
    over_road = []
    for b in buildings:
        xs, ys = rect_pts(b["x"], b["y"], b["w"], b["d"], b["ang"], 0.5)
        if hit(road, xs, ys):
            over_road.append(b["id"])
    rep["buildings_on_road"] = len(over_road)
    over_river = [b["id"] for b in buildings
                  if hit(river_m, *rect_pts(b["x"], b["y"], b["w"], b["d"], b["ang"], 0.5))]
    rep["buildings_in_river"] = len(over_river)
    # bina-bina çakışması (eksen hizalı yaklaşık + dönük için köşe testi)
    ov = 0
    for i, a in enumerate(buildings):
        for b in buildings[i + 1:]:
            if abs(a["x"] - b["x"]) > 40 or abs(a["y"] - b["y"]) > 40:
                continue
            xs, ys = rect_pts(a["x"], a["y"], a["w"] - .2, a["d"] - .2, a["ang"], 0.7)
            ca, sa_ = math.cos(-b["ang"]), math.sin(-b["ang"])
            u = (xs - b["x"]) * ca - (ys - b["y"]) * sa_
            v = (xs - b["x"]) * sa_ + (ys - b["y"]) * ca
            if np.any((abs(u) < b["w"] / 2 - .1) & (abs(v) < b["d"] / 2 - .1)):
                ov += 1
    rep["building_overlaps"] = ov
    rep["by_type"] = {}
    for b in buildings:
        rep["by_type"][b["type"]] = rep["by_type"].get(b["type"], 0) + 1
    # her bina bir yola ≤ 14 m mesafede mi (erişim)
    ys_, xs_ = np.nonzero(road)
    pts = np.stack([xs_ / RES, ys_ / RES], 1)[::3]
    far = []
    for b in buildings:
        if b["type"] in ("farmhouse", "torii", "radio_mast", "lookout", "honden"):
            continue
        dmin = np.min(np.hypot(pts[:, 0] - b["x"], pts[:, 1] - b["y"]))
        if dmin > 16:
            far.append(b["id"])
    rep["buildings_far_from_road_16m"] = len(far)
    rep["lamps"], rep["poles"], rep["vending"] = len(lamps), len(poles), len(vend)
    return rep, set(far)

report, far_ids = qa()
# yola uzak bina: sil (erişimi olmayan ev gerçek planda olmaz)
buildings = [b for b in buildings if b["id"] not in far_ids]
report, _ = qa()
print(json.dumps(report, indent=1, ensure_ascii=False))

# ------------------------------------------------------------ çizim
S = PX
img = Image.new("RGB", (int(W * S), int(H * S)), (238, 232, 214))
dr = ImageDraw.Draw(img, "RGBA")
INK = (40, 36, 30)
PAPER = (238, 232, 214)
try:
    F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    FT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except Exception:
    F = FB = FT = ImageFont.load_default()

def P(x, y):
    return (x * S, y * S)

def poly_px(pts):
    return [P(x, y) for x, y in pts]

# tarlalar: çeltik/ekin parselleri (pirinç tarlası düz teras, sıra çizgili)
def draw_field(x0, y0, x1, y1):
    cw = rnd.choice([14, 16, 18])
    ch = rnd.choice([10, 12])
    x = x0
    while x < x1:
        y = y0
        while y < y1:
            xe, ye = min(x + cw, x1), min(y + ch, y1)
            sub = np.zeros((GH, GW), bool)
            cxm, cym = (x + xe) / 2, (y + ye) / 2
            xs, ys = rect_pts(cxm, cym, xe - x, ye - y, 0, 1.0)
            if not hit(road, xs, ys) and not hit(solid, xs, ys) and (xe - x) > 4 and (ye - y) > 4:
                tone = rnd.choice([(214, 222, 190, 255), (205, 215, 178, 255), (222, 218, 180, 255)])
                dr.rectangle([P(x + .5, y + .5), P(xe - .5, ye - .5)], fill=tone, outline=(120, 112, 90))
                kind = rnd.random()
                if kind < .55:   # sıra ekin
                    yy = y + 2
                    while yy < ye - 1:
                        dr.line([P(x + 1, yy), P(xe - 1, yy)], fill=(150, 160, 118), width=1)
                        yy += 1.6
                else:            # su dolu çeltik
                    dr.line([P(x + 2, cym), P(xe - 2, cym)], fill=(168, 190, 196), width=1)
            y += ch
        x += cw

for f in FIELDS:
    if f[0] in (112, 228):
        continue
    draw_field(*f)

# orman: ağaçlar (üçgen sugi), bölgeye ve boş alana
trees = []
for _ in range(9000):
    x, y = rnd.uniform(0, W), rnd.uniform(0, H)
    if not inside(forest_z, x, y):
        continue
    if inside(road, x, y) or inside(solid, x, y) or inside(lots, x, y) or inside(field, x, y):
        continue
    if any((x - tx) ** 2 + (y - ty) ** 2 < 5.0 for tx, ty in trees[-120:]):
        continue
    trees.append((x, y))
# orman ağaçları tapınak/avlu içine girmesin: lots/solid kontrol edildi
# kasaba içi boşluklar: arka bahçe ağaçları, ağaç sıraları
for _ in range(900):
    x, y = rnd.uniform(100, 336), rnd.uniform(130, 330)
    if inside(road, x, y) or inside(solid, x, y) or inside(lots, x, y) or inside(field, x, y) \
            or inside(forest_z, x, y):
        continue
    if rnd.random() < .16:
        trees.append((x, y))
# batı yamacı, kuzey: nehir dışı kalan yeşil
for x, y in trees:
    h = 5.5 if inside(forest_z, x, y) else 4
    r = 1.7 if inside(forest_z, x, y) else 2.1
    if inside(forest_z, x, y):
        dr.polygon([P(x, y - h / 2 - 1), P(x - r, y + h / 2), P(x + r, y + h / 2)],
                   fill=(205, 212, 188), outline=(70, 80, 60))
    else:
        dr.ellipse([P(x - r, y - r), P(x + r, y + r)], fill=(195, 208, 170), outline=(80, 92, 62))

# otopark/avlu zeminleri
for a in areas:
    x0, y0, x1, y1 = a["rect"]
    col = {"parking": (205, 202, 192), "park": (200, 215, 175), "schoolyard": (225, 215, 188),
           "shrine_court": (214, 206, 188), "temple_court": (214, 206, 188),
           "cemetery": (210, 208, 190), "station_plaza": (214, 210, 198), "steps": (225, 220, 205)}.get(a["type"], (215, 210, 195))
    dr.rectangle([P(x0, y0), P(x1, y1)], fill=col + (255,), outline=INK)
    if a["type"] == "parking":
        cols = int((x1 - x0) / 2.7)
        for c in range(cols):
            xx = x0 + 1 + c * 2.7
            dr.line([P(xx, y0 + 1), P(xx, y0 + 6)], fill=(110, 105, 95), width=1)
            dr.line([P(xx, y1 - 6), P(xx, y1 - 1)], fill=(110, 105, 95), width=1)
    if a["type"] == "cemetery":
        yy = y0 + 3
        while yy < y1 - 2:
            xx = x0 + 3
            while xx < x1 - 2:
                dr.rectangle([P(xx, yy), P(xx + 1.0, yy + 1.4)], fill=(150, 148, 140), outline=INK)
                xx += 3.2
            yy += 3.6
    if a["type"] == "steps":
        yy = a["rect"][1]
        while yy < a["rect"][3]:
            dr.line([P(a["rect"][0], yy), P(a["rect"][2], yy)], fill=INK, width=1)
            yy += 0.9
    if a["type"] == "park":
        dr.ellipse([P((x0 + x1) / 2 - 2, (y0 + y1) / 2 - 2), P((x0 + x1) / 2 + 2, (y0 + y1) / 2 + 2)], outline=INK)

# nehir
dr.line(poly_px(RIVER), fill=(188, 210, 220), width=int(RIVER_W * S))
dr.line(poly_px([(x - RIVER_W / 2, y) for x, y in RIVER]), fill=INK, width=2)
dr.line(poly_px([(x + RIVER_W / 2, y) for x, y in RIVER]), fill=INK, width=2)
for ll in (LEVEE_L, LEVEE_R):
    dr.line(poly_px(ll), fill=(110, 104, 90), width=1)
    pts = walk(ll, 4)
    for x, y, a in pts:
        dr.line([P(x, y), P(x + math.sin(a) * 1.0 * -1 if False else x, y + 1.0)], fill=(110, 104, 90), width=1)
# su dalgası
for x, y, a in walk(RIVER, 11):
    dr.arc([P(x - 2, y - 1), P(x, y + 1)], 180, 360, fill=(90, 120, 140))
    dr.arc([P(x, y - 1), P(x + 2, y + 1)], 0, 180, fill=(90, 120, 140))

# kanal (batı tarla kenarı)
dr.line(poly_px([(22, 150), (22, 330)]), fill=(188, 210, 220), width=int(2 * S))
labels.append((22, 146, "canal"))

# yollar: önce koyu dış çizgi, sonra kağıt rengi iç
STYLE = {"road": (INK, PAPER), "alley": (INK, PAPER), "dirt": ((90, 82, 66), (226, 214, 184)),
         "trail": ((90, 82, 66), (238, 232, 214))}
for name, kind, wd, pts in ROADS:
    if len(pts) < 2 or name == "Kamiyama Rd Ext":
        continue
    o, i = STYLE[kind]
    wpx = int(wd * S)
    dr.line(poly_px(pts), fill=o, width=wpx + 3, joint="curve")
for name, kind, wd, pts in ROADS:
    if len(pts) < 2 or name == "Kamiyama Rd Ext":
        continue
    o, i = STYLE[kind]
    dr.line(poly_px(pts), fill=i, width=int(wd * S) - 1, joint="curve")
    if kind in ("dirt", "trail"):
        for x, y, a in walk(pts, 3):
            dr.point(P(x, y), fill=o)
    if kind == "road":
        # orta şerit kesikli
        for x, y, a in walk(pts, 5):
            dr.line([P(x, y), P(x + math.cos(a) * 2.2, y + math.sin(a) * 2.2)], fill=(190, 182, 160), width=1)
# köprü (Old Kaido nehir üstü)
bx0, bx1 = 64, 96
dr.rectangle([P(bx0, 218 - 3.6), P(bx1, 218 + 3.6)], fill=PAPER, outline=INK)
dr.line([P(bx0, 218 - 3.6), P(bx1, 218 - 3.6)], fill=INK, width=3)
dr.line([P(bx0, 218 + 3.6), P(bx1, 218 + 3.6)], fill=INK, width=3)
# demiryolu
dr.line([P(0, RAIL_Y), P(W, RAIL_Y)], fill=INK, width=2)
dr.line([P(0, RAIL_Y + 2), P(W, RAIL_Y + 2)], fill=INK, width=2)
for x in np.arange(0, W, 1.4):
    dr.line([P(x, RAIL_Y - .4), P(x, RAIL_Y + 2.4)], fill=INK, width=1)
dr.rectangle([P(*(203, RAIL_Y + 3)), P(247, RAIL_Y + 6.5)], fill=(225, 218, 200), outline=INK)  # peron
labels.append((150, RAIL_Y + 12, "<- to Terminal City"))
labels.append((355, RAIL_Y + 12, "to Nagiri Jct. (22 km) ->"))
labels.append((22, 214, "To Kamiyama (6 km)"))

# binalar
FILL = {"house": (232, 226, 206), "machiya": (226, 218, 196), "apartment": (222, 216, 200),
        "shop": (240, 226, 190), "warehouse": (214, 210, 200), "abandoned": (205, 200, 190),
        "farmhouse": (232, 226, 206)}
SPECIAL = (240, 232, 200)
for b in buildings:
    c = rect_corners(b["x"], b["y"], b["w"], b["d"], b["ang"])
    fill = FILL.get(b["type"], SPECIAL)
    if b["type"] == "torii":
        dr.line([P(*c[0]), P(*c[1])], fill=(150, 40, 30), width=3)
        continue
    dr.polygon(poly_px(c), fill=fill + (255,), outline=INK)
    # çatı sırtı: uzun eksen boyunca
    ang = b["ang"]
    ux, uy = math.cos(ang), math.sin(ang)
    L = b["w"] / 2 - .4
    if b["w"] < b["d"]:
        ux, uy, L = -math.sin(ang), math.cos(ang), b["d"] / 2 - .4
    dr.line([P(b["x"] - ux * L, b["y"] - uy * L), P(b["x"] + ux * L, b["y"] + uy * L)],
            fill=(110, 100, 85), width=1)
    if b["type"] == "abandoned":
        dr.line([P(*c[0]), P(*c[2])], fill=(150, 90, 70), width=1)
    if b["floors"] >= 2:
        dr.ellipse([P(b["x"] - .6, b["y"] - .6), P(b["x"] + .6, b["y"] + .6)], outline=INK)

# lamba, direk, otomat
for x, y in lamps:
    dr.ellipse([P(x - .8, y - .8), P(x + .8, y + .8)], fill=(250, 214, 110), outline=INK)
for x, y in poles:
    dr.ellipse([P(x - .45, y - .45), P(x + .45, y + .45)], fill=INK)
for x, y in vend:
    dr.rectangle([P(x - .5, y - .5), P(x + .5, y + .5)], fill=(190, 70, 60), outline=INK)

# etiketler
for x, y, t in labels:
    tw = dr.textlength(t, font=F)
    dr.text((x * S - tw / 2, y * S - 5), t, font=F, fill=INK, stroke_width=2, stroke_fill=PAPER + (255,))
    dr.text((x * S - tw / 2, y * S - 5), t, font=F, fill=INK)
for name, kind, wd, pts in ROADS:
    if kind == "road" and polylen(pts) > 60:
        x, y, a = walk(pts, polylen(pts) / 2)[1] if len(walk(pts, polylen(pts) / 2)) > 1 else walk(pts, 1)[0]
        dr.text((x * S - 25, y * S - 4), name.upper(), font=F, fill=(120, 40, 30), stroke_width=2, stroke_fill=PAPER + (255,))
        dr.text((x * S - 25, y * S - 4), name.upper(), font=F, fill=(120, 40, 30))
dr.text((14 * S, 14 * S), "MIZUNAGI-CHO", font=FT, fill=INK)
dr.text((14 * S, 14 * S + 26), "Town plan 1991  |  440 x 352 m  |  1 px = 0.4 m", font=F, fill=INK)
labels_x = [("SUGI FOREST", 300, 18), ("FARMLAND", 395, 300), ("FARMLAND", 22, 240)]
for t, x, y in labels_x:
    dr.text((x * S, y * S), t, font=FB, fill=(70, 80, 60), stroke_width=2, stroke_fill=PAPER + (255,))
    dr.text((x * S, y * S), t, font=FB, fill=(70, 80, 60))
# ölçek çubuğu ve kuzey
dr.rectangle([P(14, 336), P(34, 338)], fill=INK)
dr.text((14 * S, 340 * S), "0   20 m", font=F, fill=INK)
dr.text((420 * S, 14 * S), "N ^", font=FB, fill=INK)
dr.rectangle([0, 0, img.width - 1, img.height - 1], outline=INK, width=3)

img.save(os.path.join(OUT, "mizunagi_plan.png"))
json.dump(dict(scale_m_per_px=1 / PX, size_m=[W, H], buildings=buildings,
               roads=[dict(name=n, kind=k, width=w, points=p) for n, k, w, p in ROADS],
               areas=areas, river=dict(points=RIVER, width=RIVER_W), fields=FIELDS,
               lamps=lamps, utility_poles=poles, vending=vend, trees=len(trees), qa=report),
          open(os.path.join(OUT, "mizunagi_plan.json"), "w"), ensure_ascii=False, indent=1)
print("tamam ->", OUT)
