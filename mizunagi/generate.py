#!/usr/bin/env python3
"""Mizunagi-cho kasaba planı üreticisi (v2: referans haritaya sadık, sıkışık Japon dokusu).

Yol ağı ve yapıların yeri referans çizimden (2000x1600 px, çerçeve 70..1930) alınır,
gerçek metreye çevrilir: 1 ref px = 0.2366 m  ->  harita 440 x 345 m.

Yerleşim kuralları (gerçek Japon kasabası):
  * evler sokağa cepheli, cephe 6-9 m, derinlik 7-10 m, aralarında 0.3-1.3 m
  * yol kenarından 0.4-1.8 m geri, arkaya 2.-3. sıra, boşluklara depo/kulübe
  * Furumachi'de dar uzun machiya, shotengai'de bitişik dükkânlar
  * bina hiçbir koşulda yola / nehre / başka binaya taşmaz (üretimde test edilir)
  * boş kalan her yer ya Sugi ormanı, ya çeltik/ekin tarlası, ya bostan

Çalıştır: python3 generate.py   ->  out/mizunagi_plan.png, out/mizunagi_plan.json
"""
import json, math, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

K = 440.0 / 1860.0                       # m / ref px
W, H = 440.0, 1460 * K                   # 440 x 345.4 m
RES = 2.0                                # çarpışma ızgarası hücre/m
GW, GH = int(W * RES) + 1, int(H * RES) + 1
S = 4.0                                  # çizim px/m
rnd = random.Random(1991)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)


def M(p):
    return ((p[0] - 70) * K, (p[1] - 70) * K)


def MP(pts):
    return [M(p) for p in pts]


# ------------------------------------------------------------ geometri
def smooth(pts, n=6):
    """Catmull-Rom: kontrol noktalarından organik eğri."""
    if len(pts) < 3:
        return list(pts)
    P = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                                    + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in (0, 1)))
    out.append(pts[-1])
    return out


def resample(pts, step):
    """(x, y, açı, yay uzunluğu) listesi."""
    out, s, acc = [], 0.0, 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1e-9:
            continue
        a = math.atan2(y1 - y0, x1 - x0)
        while acc <= L:
            out.append((x0 + (x1 - x0) * acc / L, y0 + (y1 - y0) * acc / L, a, s + acc))
            acc += step
        acc -= L
        s += L
    return out


def rect_corners(cx, cy, w, d, a):
    c, s_ = math.cos(a), math.sin(a)
    return [(cx + u * c - v * s_, cy + u * s_ + v * c) for u, v in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2))]


def rect_samples(cx, cy, w, d, a, step=0.5):
    u, v = np.meshgrid(np.arange(-w / 2, w / 2 + 1e-6, step), np.arange(-d / 2, d / 2 + 1e-6, step))
    c, s_ = math.cos(a), math.sin(a)
    return cx + u * c - v * s_, cy + u * s_ + v * c


# ------------------------------------------------------------ ızgaralar
def grid():
    return np.zeros((GH, GW), bool)


road, river, solid, reserved, settle, fieldz, forestz = (grid() for _ in range(7))


def gidx(xs, ys):
    return (np.clip((ys * RES).astype(int), 0, GH - 1), np.clip((xs * RES).astype(int), 0, GW - 1))


def any_hit(mask, xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    iy, ix = gidx(xs, ys)
    out = (xs < 0.5) | (xs > W - .5) | (ys < .5) | (ys > H - .5)
    return bool(mask[iy, ix].any() or out.any())


def all_in(mask, xs, ys):
    iy, ix = gidx(np.asarray(xs, float), np.asarray(ys, float))
    return bool(mask[iy, ix].all())


def at(mask, x, y):
    return bool(mask[min(GH - 1, max(0, int(y * RES))), min(GW - 1, max(0, int(x * RES)))])


def stamp_line(mask, pts, width):
    r = width / 2 * RES
    for x, y, _, _ in resample(pts, 0.4):
        cx, cy = x * RES, y * RES
        x0, x1 = max(0, int(cx - r) - 1), min(GW, int(cx + r) + 2)
        y0, y1 = max(0, int(cy - r) - 1), min(GH, int(cy + r) + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (xx + .5 - cx) ** 2 + (yy + .5 - cy) ** 2 <= r * r


def stamp_poly(mask, poly, val=True):
    im = Image.new("1", (GW, GH), 0)
    ImageDraw.Draw(im).polygon([(x * RES, y * RES) for x, y in poly], fill=1)
    a = np.array(im, bool)
    if val:
        mask |= a
    else:
        mask &= ~a


def stamp_rect(mask, cx, cy, w, d, a, val=True):
    stamp_poly(mask, rect_corners(cx, cy, w, d, a), val)


# ------------------------------------------------------------ yükselti (kontur için)
yy, xx = np.mgrid[0:GH, 0:GW] / RES
noise = np.zeros_like(xx)
for f, amp in ((0.021, 1.6), (0.047, 0.8), (0.11, 0.3)):
    ph = rnd.random() * 6.28
    noise += amp * np.sin(xx * f + ph) * np.cos(yy * f * 1.3 + ph * 0.7)
ridge = np.clip((140 - yy) / 120, 0, 1) ** 1.15 * 30          # kuzey yamaç
east = np.clip((xx - 300) / 140, 0, 1) * np.clip((250 - yy) / 200, 0, 1) * 14
west = np.clip((60 - xx) / 60, 0, 1) * np.clip((150 - yy) / 150, 0, 1) * 8
HEIGHT = ridge + east + west + noise * np.clip((190 - yy) / 120, 0.15, 1)

# ------------------------------------------------------------ nehir, kanal, demiryolu
RIVER = smooth(MP([(395, 60), (420, 220), (470, 370), (468, 480), (410, 620), (345, 760), (330, 880), (378, 965),
                   (322, 1060), (310, 1150), (360, 1270), (400, 1380), (402, 1545)]))
RIVER_W = 14.0
stamp_line(river, RIVER, RIVER_W)
stamp_line(solid, RIVER, RIVER_W + 9)                    # set + set yolu payı
CANAL = smooth(MP([(150, 600), (140, 760), (136, 1000), (132, 1250), (175, 1385)]))
stamp_line(river, CANAL, 2.4)
stamp_line(solid, CANAL, 3.6)
RAIL_Y = M((0, 1392))[1]
stamp_line(road, [(0, RAIL_Y), (W, RAIL_Y)], 6.0)

# ------------------------------------------------------------ yollar (ref px)
# tür: main 7 m, road 6 m, lane 4-5 m, alley (roji) 2.7-3 m, dirt 3.6 m, trail 3 m
ROADS_REF = [
    ("Old Kaido (west)", "road", 6.0, [(60, 975), (200, 975), (325, 972)]),
    ("Old Kaido", "road", 6.5, [(432, 968), (600, 966), (800, 964), (990, 962)]),
    ("Chuo-dori", "main", 7.0, [(1005, 958), (1006, 1150), (1004, 1392), (1004, 1545)]),
    ("Sando", "lane", 5.0, [(990, 962), (990, 800), (978, 680), (965, 625), (950, 600), (946, 556)]),
    ("Mountain Road", "road", 6.0, [(1010, 965), (1100, 890), (1190, 800), (1330, 715), (1430, 688), (1510, 630), (1570, 588),
                                     (1650, 520), (1750, 492), (1850, 508), (1898, 470), (1895, 400), (1860, 330), (1858, 275),
                                     (1890, 240), (1840, 200)]),
    ("East Road", "road", 6.5, [(1006, 1086), (1150, 1084), (1300, 1080), (1450, 1074), (1600, 1064), (1655, 1068),
                                  (1690, 1110), (1722, 1160), (1750, 1170)]),
    ("Sakamichi", "lane", 4.5, [(1750, 1170), (1740, 1060), (1708, 1012), (1745, 960), (1800, 900), (1845, 848)]),
    ("Town Hall St", "lane", 4.5, [(1006, 978), (1100, 974), (1215, 980)]),
    ("Market St", "lane", 4.5, [(1006, 1150), (1120, 1148), (1240, 1142)]),
    ("River Road", "road", 6.0, [(455, 968), (452, 1080), (458, 1160), (485, 1240), (540, 1300), (600, 1352), (700, 1372),
                                  (850, 1372), (1004, 1368)]),
    ("Station Front", "lane", 5.0, [(1004, 1368), (1150, 1366), (1300, 1366)]),
    ("Shotengai", "lane", 5.0, [(775, 1162), (880, 1162), (1004, 1160)]),
    # roji / ara sokaklar
    ("Roji 1", "alley", 3.0, [(745, 966), (742, 1060), (738, 1160)]),
    ("Roji 2", "alley", 3.0, [(800, 966), (798, 1100)]),
    ("Roji 3", "alley", 3.0, [(910, 966), (908, 1060)]),
    ("Roji 4", "alley", 3.0, [(800, 1105), (880, 1108), (960, 1110)]),
    ("Roji 5", "alley", 3.0, [(828, 1162), (826, 1260), (828, 1372)]),
    ("Roji 6", "alley", 3.0, [(965, 1160), (963, 1260), (965, 1368)]),
    ("Roji 7", "alley", 3.0, [(458, 1118), (600, 1114), (738, 1110)]),
    ("Roji 8", "alley", 3.0, [(560, 966), (562, 1114)]),
    ("Roji 9", "alley", 3.0, [(640, 1114), (650, 1230), (690, 1300), (700, 1372)]),
    ("Roji 10", "alley", 3.0, [(1100, 1150), (1098, 1260), (1100, 1366)]),
    ("Roji 11", "alley", 3.0, [(1220, 1145), (1215, 1250), (1220, 1366)]),
    ("Roji 12", "alley", 3.0, [(1006, 1260), (1100, 1260), (1215, 1252), (1330, 1258)]),
    ("Roji 13", "alley", 3.0, [(1250, 1045), (1252, 1082)]),
    ("Roji 14", "alley", 3.0, [(1060, 965), (1130, 905), (1250, 900), (1330, 905)]),
    ("Roji 15", "alley", 3.0, [(1420, 1076), (1430, 1200), (1460, 1260)]),
    ("Roji 16", "alley", 3.0, [(1300, 1081), (1300, 1150), (1320, 1172)]),
    # Furumachi
    ("Furumachi D", "lane", 4.0, [(598, 966), (596, 880), (596, 800), (600, 700)]),
    ("Furumachi A", "lane", 4.0, [(596, 800), (700, 796), (792, 786)]),
    ("Furumachi B", "lane", 4.0, [(805, 690), (800, 790), (778, 880), (768, 964)]),
    ("Furumachi C", "lane", 4.0, [(800, 792), (900, 784), (985, 780)]),
    ("Well alley", "alley", 2.7, [(812, 825), (880, 830), (945, 838)]),
    ("Furumachi E", "alley", 2.7, [(470, 966), (480, 860), (520, 760), (600, 700)]),
    ("Furumachi F", "alley", 2.7, [(596, 880), (690, 878), (772, 876)]),
    # toprak / patika
    ("Furumachi path", "dirt", 3.6, [(520, 760), (590, 662), (700, 655), (830, 645), (950, 625)]),
    ("Meadow path", "dirt", 3.6, [(978, 665), (1200, 670), (1430, 690)]),
    ("West Bank Lane", "dirt", 3.6, [(215, 600), (205, 800), (210, 975), (200, 1100), (205, 1300), (195, 1380)]),
    ("Canal path", "dirt", 3.6, [(110, 782), (205, 782)]),
    ("SE farm road", "dirt", 3.6, [(1555, 1345), (1600, 1302), (1700, 1310), (1800, 1322)]),
    ("Fire lane", "dirt", 3.6, [(1650, 1150), (1600, 1220), (1555, 1270), (1555, 1345)]),
    ("Levee path E", "trail", 2.8, [(372, 640), (395, 700), (370, 800), (400, 900), (440, 965)]),
    ("Levee path S", "trail", 2.8, [(440, 975), (415, 1080), (385, 1170), (420, 1270), (450, 1385)]),
    ("Bike trail", "trail", 3.0, [(1240, 1142), (1320, 1170), (1400, 1205), (1500, 1240), (1555, 1270)]),
    ("Forest link", "trail", 3.0, [(1500, 640), (1560, 560), (1620, 470), (1690, 370)]),
    ("Mountain houses path", "trail", 3.0, [(1100, 352), (1180, 380), (1250, 425), (1350, 390), (1450, 378), (1530, 342),
                                             (1620, 330), (1700, 312), (1790, 345), (1858, 300)]),
]
ROADS = [(n, k, w, smooth(MP(p))) for n, k, w, p in ROADS_REF]
ROADS.append(("Kaido bridge", "bridge", 6.0, MP([(318, 972), (440, 968)])))
for n, k, w, pts in ROADS:
    stamp_line(road, pts, w + 0.6)

# ------------------------------------------------------------ bölgeler
TOWN = MP([(405, 690), (600, 662), (965, 642), (1020, 700), (1042, 940), (1180, 830), (1330, 760), (1345, 880), (1560, 905),
           (1700, 940), (1790, 1000), (1800, 1250), (1760, 1350), (1450, 1385), (1300, 1385), (400, 1385), (405, 1250),
           (445, 1000), (430, 900)])
WESTBANK = MP([(160, 620), (300, 620), (318, 960), (290, 1100), (300, 1385), (150, 1385), (150, 620)])
stamp_poly(settle, TOWN)
stamp_poly(settle, WESTBANK)
FIELDS = [MP(p) for p in [
    [(72, 600), (128, 600), (125, 1385), (72, 1385)],                      # batı yaka çeltik
    [(72, 1408), (1928, 1408), (1928, 1528), (72, 1528)],                  # demiryolu güneyi
    [(1570, 1170), (1928, 1170), (1928, 1385), (1570, 1385)],              # güneydoğu
    [(1060, 700), (1310, 690), (1420, 700), (1280, 760), (1150, 840), (1060, 870)],   # çayır: bostan
]]
for f in FIELDS:
    stamp_poly(fieldz, f)
fieldz &= ~road & ~river
settle &= ~fieldz
# orman = ne kasaba ne tarla olan kara (kuzey, doğu yamaç, batı tepe)
forestz[:] = ~settle & ~fieldz & ~river
stamp_poly(forestz, MP([(0, 1395), (1930, 1395), (1930, 1600), (0, 1600)]), False)
stamp_poly(forestz, MP([(1700, 940), (1930, 900), (1930, 1170), (1800, 1170)]), True)

# ------------------------------------------------------------ özel yapılar
buildings, areas = [], []


def free(cx, cy, w, d, a, margin):
    xs, ys = rect_samples(cx, cy, w + 2 * margin, d + 2 * margin, a)
    return not (any_hit(road, xs, ys) or any_hit(solid, xs, ys) or any_hit(river, xs, ys))


def add_building(typ, cx, cy, w, d, a, floors=1, label=None, margin=0.5, nudge=False):
    cand = [(0, 0)] + ([(r * math.cos(t * math.pi / 4), r * math.sin(t * math.pi / 4))
                        for r in (1.5, 3, 4.5, 6) for t in range(8)] if nudge else [])
    for dx, dy in cand:
        if free(cx + dx, cy + dy, w, d, a, margin):
            b = dict(id=len(buildings), type=typ, x=round(cx + dx, 2), y=round(cy + dy, 2), w=round(w, 2), d=round(d, 2),
                     ang=round(a, 4), floors=floors, label=label)
            buildings.append(b)
            stamp_rect(solid, b["x"], b["y"], w + 0.3, d + 0.3, a)
            return b
    if label:
        print("UYARI: özel yapı yerleşmedi:", typ, label)
    return None


def special(typ, ref, wpx, hpx, label=None, floors=1, deg=0.0, nudge=True, margin=0.6):
    cx, cy = M(ref)
    return add_building(typ, cx, cy, wpx * K, hpx * K, math.radians(deg), floors, label, margin, nudge)


def area(typ, x0, y0, x1, y1, label=None):
    a, b = M((x0, y0)), M((x1, y1))
    areas.append(dict(type=typ, rect=[round(a[0], 2), round(a[1], 2), round(b[0], 2), round(b[1], 2)], label=label))
    quad = [a, (b[0], a[1]), b, (a[0], b[1])]
    stamp_poly(reserved, quad)
    stamp_poly(solid, quad)


# tapınak terası (doğal sırt üstünde) ve ayrı terasta tera + mezarlık
special("honden", (945, 272), 26, 26, None, nudge=False, margin=0.3)
special("haiden", (945, 345), 56, 40, "Shrine hall", nudge=False, margin=0.3)
special("shrine_office", (993, 362), 32, 25, "Shrine office", nudge=False, margin=0.3)
special("priest_house", (1065, 335), 50, 36, "Priest's house", nudge=False)
special("temple", (795, 370), 80, 40, "Temple", nudge=False)
special("bell_tower", (818, 340), 14, 14, None, nudge=False, margin=0.2)
area("cemetery", 660, 372, 724, 436, "Cemetery")
area("steps", 937, 372, 953, 548, None)
special("torii", (946, 552), 30, 5, None, nudge=False, margin=0)
# istasyon bölgesi (demiryolunun güneyi meydan)
special("station", (930, 1432), 80, 40, "Mizunagi Station", nudge=False)
special("post_office", (790, 1432), 125, 42, "Post Office (our room upstairs)", floors=2, nudge=False)
area("platform", 900, 1400, 1120, 1410, None)
area("parking", 1040, 1435, 1180, 1500, "P  station")
special("bike_garage", (955, 1338), 42, 28, "Bike garage")
special("ja_coop", (1060, 1328), 60, 42, "JA co-op (rice store)")
special("freight_shed", (1290, 1363), 95, 26, "Freight shed", margin=0.3)
# ticaret
special("konbini", (1050, 1210), 48, 36, "Konbini (24h)")
area("parking", 1040, 1180, 1092, 1196, None)
special("ramen", (1042, 1028), 40, 30, "Ramen-ya")
special("town_hall", (1135, 1030), 78, 40, "Town Hall", floors=2)
special("police_box", (895, 1000), 24, 20, "Police box")
special("market", (1110, 1118), 52, 26, "Market")
special("barber", (955, 1135), 30, 24, "Barber")
special("bicycle_shop", (1160, 1192), 44, 30, "Bicycle shop")
special("supermarket", (1490, 1300), 92, 52, "Supermarket")
area("parking", 1360, 1272, 1440, 1335, "P  supermarket")
special("gas_stand", (1530, 1110), 50, 36, "Gas stand")
special("sento", (690, 1085), 56, 40, "Sento (bath)")
area("park", 585, 1000, 650, 1060, "Playground")
special("fire_station", (1380, 1180), 46, 32, "Fire station", floors=2)
special("clinic", (1595, 1155), 66, 44, "Clinic", floors=2)
area("parking", 1630, 1190, 1680, 1230, "P")
# okul
special("school", (1440, 935), 170, 40, "Mizunagi Primary School", floors=2, nudge=False)
special("school_wing", (1540, 905), 44, 82, None, floors=2, nudge=False)
special("gym", (1325, 920), 56, 40, "gym")
area("schoolyard", 1360, 965, 1500, 1028, None)
special("community_hall", (1600, 1000), 72, 38, "Community hall")
special("water_tower", (1868, 818), 26, 26, "Water tower", margin=0.2)
special("forestry_hut", (1495, 672), 36, 36, "Forestry hut", deg=-35)
special("lookout", (1715, 592), 30, 14, "Lookout rock", margin=0.2)
special("radio_mast", (1525, 270), 18, 18, "Radio mast", margin=0.2)
special("rest_hut", (1790, 352), 32, 22, "Old postmen's rest hut")
special("tunnel", (1820, 185), 34, 24, "Old tunnel (sealed)", margin=0)
for i, p in enumerate([(1470, 365), (1555, 400), (1395, 400), (1620, 360)]):
    special("mountain_house", p, 40, 32, "Mountain houses" if i == 1 else None, deg=rnd.uniform(-20, 20))

# ------------------------------------------------------------ sıkışık parsel yerleşimi
KIND_SIZE = {   # cephe (min,max), derinlik (min,max), kat
    "house": ((6.0, 8.5), (7.0, 9.5), 2), "old_house": ((6.5, 9.0), (7.5, 10.0), 1),
    "machiya": ((4.5, 6.0), (9.5, 12.5), 2), "shop": ((4.5, 7.0), (8.0, 11.0), 2),
    "apartment": ((9.5, 12.0), (13.0, 16.0), 3), "warehouse": ((7.5, 10.0), (8.0, 11.0), 1),
    "akiya": ((6.0, 8.0), (7.0, 9.0), 1), "farmhouse": ((9.0, 12.0), (8.5, 11.0), 1),
}
MIX = {
    "furumachi": [("machiya", .45), ("old_house", .32), ("akiya", .13), ("warehouse", .10)],
    "shop": [("shop", .70), ("machiya", .12), ("house", .12), ("apartment", .06)],
    "res": [("house", .66), ("old_house", .14), ("apartment", .06), ("akiya", .07), ("warehouse", .07)],
    "rural": [("farmhouse", .55), ("old_house", .25), ("warehouse", .20)],
}


def pick(zone):
    r, acc = rnd.random(), 0.0
    for k, p in MIX[zone]:
        acc += p
        if r <= acc:
            return k
    return MIX[zone][0][0]


SHOP_STREETS = {"Shotengai", "Chuo-dori", "Market St", "Station Front", "Town Hall St"}
FURUMACHI = {"Furumachi A", "Furumachi B", "Furumachi C", "Furumachi D", "Furumachi E", "Furumachi F", "Well alley", "Old Kaido"}
NO_FRONTAGE = {"Kaido bridge", "Canal path", "Fire lane", "Meadow path", "Furumachi path"}


def zone_of(name, x, y):
    if name in SHOP_STREETS:
        return "shop"
    if name in FURUMACHI and y < M((0, 990))[1]:
        return "furumachi"
    if x < M((330, 0))[0]:
        return "rural"
    return "res"


def frontage_rows(name, kind, wd, pts, rows):
    samples = resample(pts, 0.5)
    if len(samples) < 4:
        return
    L = samples[-1][3]
    for side in (-1, 1):
        back = [0.0] * (len(samples) + 2)     # her yay noktası için dolu sıranın arka kenarı
        for row in range(rows):
            s = rnd.uniform(0.5, 3.0)
            while s < L - 2:
                x, y, a, _ = samples[min(len(samples) - 1, int(s / 0.5))]
                z = zone_of(name, x, y)
                k = pick(z)
                (f0, f1), (d0, d1), fl = KIND_SIZE[k]
                fw, dp = rnd.uniform(f0, f1), rnd.uniform(d0, d1)
                gap = rnd.uniform(0.25, 0.5) if z in ("shop", "furumachi") else rnd.uniform(0.4, 1.0)
                sc = s + fw / 2
                if sc > L - 1:
                    break
                cx_, cy_, ca, _ = samples[min(len(samples) - 1, int(sc / 0.5))]
                i0, i1 = int(s / 0.5), min(len(back) - 1, int((s + fw) / 0.5))
                if row == 0:
                    setb = rnd.uniform(0.4, 0.9) if z in ("shop", "furumachi") else rnd.uniform(0.8, 1.8)
                    off = wd / 2 + 0.3 + setb + dp / 2
                else:
                    prev = max(back[i0:i1 + 1] or [0])
                    if prev <= 0:
                        s += 2.0
                        continue
                    off = prev + rnd.uniform(0.6, 1.2) + dp / 2
                ang = ca + math.radians(rnd.uniform(-2.5, 2.5))
                nx, ny = -math.sin(ca) * side, math.cos(ca) * side
                bx, by = cx_ + nx * off, cy_ + ny * off
                xs, ys = rect_samples(bx, by, fw, dp, ang, 1.0)
                if all_in(settle, xs, ys) and not any_hit(reserved, xs, ys):
                    b = add_building(k, bx, by, fw, dp, ang, fl if k != "house" else rnd.choice((1, 2, 2)), None, 0.3) \
                        if rnd.random() > 0.04 else None          # arada bir boş arsa
                    if b:
                        for i in range(i0, i1 + 1):
                            back[i] = max(back[i], off + dp / 2)
                        s += fw + gap
                        continue
                s += 1.0


ordered = sorted(ROADS, key=lambda r: {"main": 0, "road": 1, "lane": 2, "alley": 3, "dirt": 4}.get(r[1], 9))
for name, kind, wd, pts in ordered:
    if name in NO_FRONTAGE or kind in ("trail", "bridge"):
        continue
    frontage_rows(name, kind, wd, pts, {"main": 3, "road": 3, "lane": 3, "alley": 2, "dirt": 1}[kind])

# boşluk dolgusu: arka bahçe deposu / kulübe ve iç evler (Japon arka bahçe dokusu)
for _ in range(16000):
    x, y = rnd.uniform(0, W), rnd.uniform(0, H)
    if not at(settle, x, y) or at(reserved, x, y) or at(solid, x, y):
        continue
    near = [b for b in buildings if abs(b["x"] - x) < 12 and abs(b["y"] - y) < 12 and b["type"] in KIND_SIZE]
    if not near:
        continue
    a = near[0]["ang"]
    if rnd.random() < .3:
        w_, d_, t = rnd.uniform(2.4, 3.6), rnd.uniform(2.4, 4.0), "shed"
    else:
        t = pick("res")
        (f0, f1), (d0, d1), _ = KIND_SIZE[t]
        w_, d_ = rnd.uniform(f0, f1), rnd.uniform(d0, d1)
    xs, ys = rect_samples(x, y, w_, d_, a, 1.0)
    if all_in(settle, xs, ys):
        add_building(t, x, y, w_, d_, a, 1, None, 0.3)

# ------------------------------------------------------------ yola erişim kontrolü (erişimsiz bina olmaz)
road_pts = np.argwhere(road)[::2] / RES          # (y, x)
keep = []
for b in buildings:
    if b["label"] or b["type"] in ("honden", "bell_tower", "torii", "tunnel", "mountain_house", "school_wing"):
        keep.append(b)
        continue
    dist = np.min(np.hypot(road_pts[:, 1] - b["x"], road_pts[:, 0] - b["y"])) - max(b["w"], b["d"]) / 2
    if dist <= (18.0 if b["type"] == "shed" else 15.0):
        keep.append(b)
    else:
        stamp_rect(solid, b["x"], b["y"], b["w"] + .3, b["d"] + .3, b["ang"], False)
buildings = keep
for i, b in enumerate(buildings):
    b["id"] = i
# yeniden: çıkarılan binaların yerine kalan ızgara
for b in buildings:
    stamp_rect(solid, b["x"], b["y"], b["w"] + .3, b["d"] + .3, b["ang"])

# ------------------------------------------------------------ sokak donatıları
lamps, poles, vending, jizo = [], [], [], [M((440, 990)), M((1010, 860))]
for name, kind, wd, pts in ROADS:
    if kind in ("trail", "bridge"):
        continue
    step = {"main": 20, "road": 24, "lane": 26, "alley": 30, "dirt": 40}[kind]
    for n, (x, y, a, _) in enumerate(resample(pts, step)[1:]):
        side = 1 if n % 2 else -1
        off = wd / 2 + 0.5
        px_, py_ = x - math.sin(a) * side * off, y + math.cos(a) * side * off
        if at(river, px_, py_) or at(solid, px_, py_):
            continue
        (lamps if kind in ("main", "road", "lane") and at(settle, px_, py_) else poles).append((round(px_, 1), round(py_, 1)))
for b in buildings:
    if b["type"] in ("konbini", "station", "shop", "gas_stand", "supermarket", "sento", "post_office") and rnd.random() < .6:
        c = rect_corners(b["x"], b["y"], b["w"] + 1.2, b["d"] + 1.2, b["ang"])[2]
        if not at(road, *c) and not at(river, *c) and not at(solid, *c):
            vending.append((round(c[0], 1), round(c[1], 1)))
BUS_STOP = M((1345, 1012))


# ------------------------------------------------------------ QA
def qa():
    over_road = over_river = overlaps = 0
    for b in buildings:
        xs, ys = rect_samples(b["x"], b["y"], b["w"] - .1, b["d"] - .1, b["ang"], 0.4)
        over_road += any_hit(road, xs, ys) and b["type"] != "tunnel"
        over_river += any_hit(river, xs, ys)
    for i, a in enumerate(buildings):
        for b in buildings[i + 1:]:
            if abs(a["x"] - b["x"]) > 25 or abs(a["y"] - b["y"]) > 25:
                continue
            xs, ys = rect_samples(a["x"], a["y"], a["w"] - .2, a["d"] - .2, a["ang"], 0.5)
            c, s_ = math.cos(-b["ang"]), math.sin(-b["ang"])
            u = (xs - b["x"]) * c - (ys - b["y"]) * s_
            v = (xs - b["x"]) * s_ + (ys - b["y"]) * c
            overlaps += bool(np.any((abs(u) < b["w"] / 2 - .1) & (abs(v) < b["d"] / 2 - .1)))
    types = {}
    for b in buildings:
        types[b["type"]] = types.get(b["type"], 0) + 1
    town_area = settle.sum() / RES ** 2
    bfoot = sum(b["w"] * b["d"] for b in buildings if at(settle, b["x"], b["y"]))
    return dict(buildings=len(buildings), buildings_on_road=int(over_road), buildings_in_water=int(over_river),
                building_overlaps=int(overlaps), building_coverage_town=round(bfoot / town_area, 3),
                lamps=len(lamps), utility_poles=len(poles), vending=len(vending), by_type=types)


report = qa()
print(json.dumps(report, ensure_ascii=False, indent=1))

# ------------------------------------------------------------ çizim
PAPER, INK, GREY, WATER = (241, 236, 222), (34, 31, 27), (150, 144, 132), (205, 222, 228)
img = Image.new("RGB", (int(W * S), int(H * S)), PAPER)


def P(x, y):
    return (x * S, y * S)


def wob(pts, amp=0.08):
    return [(x * S + rnd.uniform(-amp, amp) * S, y * S + rnd.uniform(-amp, amp) * S) for x, y in pts]


def font(names, size):
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            pass
    return ImageFont.load_default()


F_IT = font(["/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"], 17)
F_SM = font(["/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"], 14)
F_BIG = font(["/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"], 22)
F_TITLE = font(["/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"], 52)

# kontur çizgileri (2 m) — sadece kasaba dışı / yamaç; yol ve suyun üstüne çizilmez
cont = np.zeros((GH, GW), bool)
h2 = np.floor(HEIGHT / 2.0)
cont[:, 1:] |= h2[:, 1:] != h2[:, :-1]
cont[1:, :] |= h2[1:, :] != h2[:-1, :]
cont &= ~road & ~river & ((~settle) | (HEIGHT > 3))
ci = Image.fromarray((cont * 255).astype(np.uint8)).resize(img.size, Image.NEAREST)
img.paste(Image.new("RGB", img.size, (196, 189, 172)), mask=ci)
dr = ImageDraw.Draw(img)

# tarlalar: çeltik (su çizgili), bostan (sıra), yollara/binalara değmez
for f in FIELDS:
    x0, x1 = min(p[0] for p in f), max(p[0] for p in f)
    y0, y1 = min(p[1] for p in f), max(p[1] for p in f)
    cw, ch = rnd.choice([(16, 11), (14, 10), (18, 12)]) if (x1 - x0) * (y1 - y0) > 6000 else (8, 6)
    y = y0
    while y < y1:
        x = x0 + rnd.uniform(-3, 0)
        while x < x1:
            w_ = cw + rnd.uniform(-2, 2)
            xs, ys = rect_samples(x + w_ / 2, y + ch / 2, w_ - 1, ch - 1, 0, 1.0)
            if all_in(fieldz, xs, ys) and not any_hit(solid, xs, ys):
                dr.polygon(wob(rect_corners(x + w_ / 2, y + ch / 2, w_ - 0.8, ch - 0.8, 0), .1), outline=(120, 113, 98), width=2)
                if rnd.random() < .6:      # çeltik: su yüzeyi
                    for k in range(1, 4):
                        dr.line([P(x + 1.5, y + k * ch / 4), P(x + w_ - 2.5, y + k * ch / 4)], fill=(150, 175, 185), width=1)
                else:                      # ekin sıraları
                    xx_ = x + 1.5
                    while xx_ < x + w_ - 1.5:
                        dr.line([P(xx_, y + 1.2), P(xx_, y + ch - 2)], fill=(150, 150, 120), width=1)
                        xx_ += 1.3
            x += w_
        y += ch

# alanlar
for a in areas:
    x0, y0, x1, y1 = a["rect"]
    if a["type"] == "parking":
        dr.rectangle([P(x0, y0), P(x1, y1)], outline=INK, width=2)
        n = max(1, int((x1 - x0) / 2.6))
        for k in range(1, n):
            xx_ = x0 + k * (x1 - x0) / n
            dr.line([P(xx_, y0), P(xx_, y0 + min(5, (y1 - y0) / 2.2))], fill=GREY, width=1)
            if y1 - y0 > 11:
                dr.line([P(xx_, y1 - 5), P(xx_, y1)], fill=GREY, width=1)
    elif a["type"] == "cemetery":
        yy_ = y0 + 1.5
        while yy_ < y1 - 2:
            xx_ = x0 + 1.5
            while xx_ < x1 - 1.5:
                dr.rectangle([P(xx_, yy_), P(xx_ + 1.4, yy_ + 1.8)], outline=INK, width=1)
                xx_ += 2.6
            yy_ += 3.0
        dr.rectangle([P(x0, y0), P(x1, y1)], outline=INK, width=1)
    elif a["type"] == "steps":
        yy_ = y0
        while yy_ < y1:
            dr.line([P(x0, yy_), P(x1, yy_)], fill=INK, width=1)
            yy_ += 1.1
        dr.rectangle([P(x0, y0), P(x1, y1)], outline=INK, width=2)
    elif a["type"] == "schoolyard":
        dr.ellipse([P(x0 + 2, y0 + 2), P(x1 - 2, y1 - 2)], outline=INK, width=2)
        dr.ellipse([P(x0 + 6, y0 + 4), P(x1 - 6, y1 - 4)], outline=GREY, width=1)
    elif a["type"] in ("park", "platform"):
        dr.rectangle([P(x0, y0), P(x1, y1)], outline=INK, width=1 if a["type"] == "park" else 2)
        if a["type"] == "park":
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            dr.ellipse([P(cx - 1.2, cy - 1.2), P(cx + 1.2, cy + 1.2)], outline=INK, width=2)

# orman: sugi üçgenleri; bostan / kasaba boşluğu: yuvarlak ağaç kümeleri
trees = []
cell_ = 5.2
blk = road | solid | river
for gy in np.arange(0, H, cell_):
    for gx in np.arange(0, W, cell_):
        x, y = gx + rnd.uniform(0, cell_), gy + rnd.uniform(0, cell_)
        if not at(forestz, x, y) or any(at(blk, x + dx, y + dy) for dx, dy in ((0, 0), (0, -5), (-2, 0), (2, 0), (0, 1))):
            continue
        trees.append(("sugi", x, y))
for _ in range(2600):
    x, y = rnd.uniform(0, W), rnd.uniform(0, H)
    if at(blk, x, y) or at(fieldz, x, y) or at(forestz, x, y):
        continue
    xs, ys = rect_samples(x, y, 3.8, 3.8, 0, 0.7)
    if any_hit(blk, xs, ys):
        continue
    if any((x - tx) ** 2 + (y - ty) ** 2 < 14 for t, tx, ty in trees[-400:] if t == "round"):
        continue
    trees.append(("round", x, y))
for t, x, y in sorted(trees, key=lambda q: q[2]):
    if t == "sugi":
        hgt, wd_ = 5.2, 1.9
        dr.polygon([P(x, y - hgt), P(x - wd_, y), P(x + wd_, y)], fill=PAPER, outline=INK, width=2)
        dr.line([P(x - wd_ * .45, y - hgt * .4), P(x + wd_ * .45, y - hgt * .4)], fill=INK, width=1)
        dr.line([P(x, y), P(x, y + 0.7)], fill=INK, width=2)
    else:
        for dx, dy, r in ((-0.7, 0.2, 1.0), (0.7, 0.2, 1.0), (0, -0.6, 1.05)):
            dr.ellipse([P(x + dx - r, y + dy - r), P(x + dx + r, y + dy + r)], fill=PAPER, outline=INK, width=2)

# nehir: su, set şevi taraması, dalgalar
dr.line([P(*p) for p in RIVER], fill=WATER, width=int(RIVER_W * S), joint="curve")
for sgn in (-1, 1):
    bank = []
    for x, y, a, s_ in resample(RIVER, 1.0):
        nx, ny = -math.sin(a) * sgn, math.cos(a) * sgn
        bank.append((x + nx * RIVER_W / 2, y + ny * RIVER_W / 2))
        if int(s_) % 3 == 0:     # set şevi tarama (legend: RIVER LEVEE)
            dr.line([P(x + nx * (RIVER_W / 2 + .3), y + ny * (RIVER_W / 2 + .3)),
                     P(x + nx * (RIVER_W / 2 + 2.0), y + ny * (RIVER_W / 2 + 2.0))], fill=GREY, width=1)
    dr.line([P(*p) for p in bank], fill=INK, width=3)
for x, y, a, s_ in resample(RIVER, 7.0):
    xx_ = x - math.sin(a) * (2.2 if int(s_ / 7) % 2 else -2.2)
    dr.arc([P(xx_ - 1.6, y - .8), P(xx_, y + .8)], 180, 360, fill=(80, 110, 125), width=2)
    dr.arc([P(xx_, y - .8), P(xx_ + 1.6, y + .8)], 0, 180, fill=(80, 110, 125), width=2)
dr.line([P(*p) for p in CANAL], fill=WATER, width=int(2.4 * S))
for sgn in (-1, 1):
    dr.line([P(x - math.sin(a) * sgn * 1.2, y + math.cos(a) * sgn * 1.2) for x, y, a, _ in resample(CANAL, 1.0)], fill=INK, width=2)

# yollar: çift çizgi (kenar mürekkep, iç kağıt); toprak yol noktalı; patika kesik
paved = [r for r in ROADS if r[1] in ("main", "road", "lane", "alley")]
for n, k, w, pts in paved:
    dr.line([P(*p) for p in pts], fill=INK, width=int(w * S) + 5, joint="curve")
for n, k, w, pts in paved:
    dr.line([P(*p) for p in pts], fill=PAPER, width=int(w * S) - 1, joint="curve")
for n, k, w, pts in ROADS:
    if k == "dirt":
        for sgn in (-1, 1):
            for x, y, a, _ in resample(pts, 1.4):
                ex, ey = x - math.sin(a) * sgn * w / 2, y + math.cos(a) * sgn * w / 2
                if not at(solid, ex, ey) and not at(river, ex, ey):
                    dr.ellipse([P(ex - .18, ey - .18), P(ex + .18, ey + .18)], fill=INK)
    if k == "trail":
        for i, (x, y, a, _) in enumerate(resample(pts, 1.2)):
            for sgn in (-1, 1):
                ex, ey = x - math.sin(a) * sgn * w / 2, y + math.cos(a) * sgn * w / 2
                if i % 3 != 2 and not at(river, ex, ey):
                    dr.line([P(ex, ey), P(ex + math.cos(a) * .8, ey + math.sin(a) * .8)], fill=INK, width=2)
# köprü
b0, b1 = M((318, 972)), M((440, 968))
dr.polygon([P(b0[0], b0[1] - 3.3), P(b1[0], b1[1] - 3.3), P(b1[0], b1[1] + 3.3), P(b0[0], b0[1] + 3.3)], fill=PAPER)
for off in (-3.6, 3.6):
    dr.line([P(b0[0], b0[1] + off), P(b1[0], b1[1] + off)], fill=INK, width=6)
# demiryolu + demiryolu köprüsü
for off in (-1.0, 1.0):
    dr.line([P(0, RAIL_Y + off), P(W, RAIL_Y + off)], fill=INK, width=3)
x = 0.0
while x < W:
    dr.line([P(x, RAIL_Y - 1.9), P(x, RAIL_Y + 1.9)], fill=INK, width=2)
    x += 2.6
rb = M((368, 1392))[0]
dr.rectangle([P(rb, RAIL_Y - 2.6), P(rb + 17, RAIL_Y + 2.6)], fill=PAPER, outline=INK, width=3)
for k in range(1, 5):
    dr.line([P(rb + k * 3.4, RAIL_Y - 2.6), P(rb + k * 3.4, RAIL_Y + 2.6)], fill=INK, width=2)

# binalar: arka parsel çizgisi (gri), gövde, mahya çizgisi
SYMBOLS = ("torii", "tunnel", "water_tower", "radio_mast")
for b in buildings:
    if b["type"] in KIND_SIZE or b["type"] == "shed":     # parsel sınırı (ref'teki gri çerçeve)
        pc = rect_corners(b["x"] + rnd.uniform(-.5, .5), b["y"] + rnd.uniform(-.5, .5), b["w"] + 1.2, b["d"] + 1.2, b["ang"])
        dr.polygon([P(*p) for p in pc], outline=(178, 171, 158), width=1)
for b in buildings:
    x, y = b["x"], b["y"]
    if b["type"] == "torii":
        c = rect_corners(x, y, b["w"], b["d"], b["ang"])
        dr.line([P(*c[0]), P(*c[1])], fill=INK, width=5)
        dr.line([P(*c[3]), P(*c[2])], fill=INK, width=3)
        continue
    if b["type"] == "tunnel":
        dr.pieslice([P(x - 3, y - 3), P(x + 3, y + 3)], 180, 360, fill=INK)
        dr.arc([P(x - 4, y - 4), P(x + 4, y + 4)], 180, 360, fill=INK, width=3)
        continue
    if b["type"] == "water_tower":
        dr.ellipse([P(x - 2.6, y - 2.6), P(x + 2.6, y + 2.6)], fill=PAPER, outline=INK, width=3)
        continue
    if b["type"] == "radio_mast":
        dr.polygon([P(x, y - 6), P(x - 1.6, y + 2), P(x + 1.6, y + 2)], outline=INK, width=2)
        continue
    c = rect_corners(x, y, b["w"], b["d"], b["ang"])
    dr.polygon(wob(c, .06), fill=PAPER, outline=INK, width=3 if b["label"] else 2)
    a = b["ang"] if b["w"] >= b["d"] else b["ang"] + math.pi / 2
    L = max(b["w"], b["d"]) / 2 - 0.9
    if b["type"] != "shed":
        dr.line([P(x - math.cos(a) * L, y - math.sin(a) * L), P(x + math.cos(a) * L, y + math.sin(a) * L)], fill=INK, width=2)
    if b["type"] == "akiya":     # terk edilmiş: kırık çatı
        dr.line([P(*c[0]), P(*c[2])], fill=GREY, width=1)
    if b["type"] in ("apartment", "school", "school_wing", "supermarket"):
        k_ = 3 if b["type"] == "apartment" else 6
        for i in range(1, k_):
            t = i / k_
            p0 = (c[0][0] + (c[1][0] - c[0][0]) * t, c[0][1] + (c[1][1] - c[0][1]) * t)
            p1 = (c[3][0] + (c[2][0] - c[3][0]) * t, c[3][1] + (c[2][1] - c[3][1]) * t)
            dr.line([P(*p0), P(*p1)], fill=GREY, width=1)

# donatılar
for x, y in poles:
    dr.ellipse([P(x - .35, y - .35), P(x + .35, y + .35)], fill=INK)
for x, y in lamps:
    dr.ellipse([P(x - .55, y - .55), P(x + .55, y + .55)], fill=(250, 214, 120), outline=INK, width=1)
for x, y in vending:
    dr.rectangle([P(x - .5, y - .4), P(x + .5, y + .4)], fill=(196, 72, 58), outline=INK)
for x, y in jizo:
    dr.ellipse([P(x - .6, y - 1.2), P(x + .6, y)], outline=INK, width=2)
    dr.line([P(x - .8, y + .2), P(x + .8, y + .2)], fill=INK, width=2)
dr.rectangle([P(BUS_STOP[0] - .6, BUS_STOP[1] - .6), P(BUS_STOP[0] + .6, BUS_STOP[1] + .6)], fill=INK)


_boxes = []


def text(x, y, t, f=F_IT):
    """Etiket; başka etiketle çakışırsa aşağı/yukarı kaydırılır."""
    tw = dr.textlength(t, font=f)
    for dy in (0, 5, -5, 10, -10, 15):
        bx = (x * S - tw / 2, (y + dy) * S - f.size / 2, x * S + tw / 2, (y + dy) * S + f.size / 2)
        if not any(bx[0] < o[2] and o[0] < bx[2] and bx[1] < o[3] and o[1] < bx[3] for o in _boxes):
            break
    _boxes.append(bx)
    dr.text(bx[:2], t, font=f, fill=INK, stroke_width=4, stroke_fill=PAPER)


for b in buildings:
    if b["label"] and b["type"] != "tunnel":
        text(b["x"], b["y"] - b["d"] / 2 - 2.6, b["label"])
text(*M((1830, 150)), "Old tunnel (sealed)")
text(*M((905, 460)), "108 steps  +16 m", F_SM)
for a in areas:
    if a["label"]:
        x0, y0, x1, y1 = a["rect"]
        text((x0 + x1) / 2, (y0 + y1) / 2, a["label"], F_SM)
for n, k, w, pts in ROADS:
    if n in ("Old Kaido", "Chuo-dori", "East Road", "River Road", "Mountain Road", "Shotengai", "Sakamichi", "Station Front"):
        smp = resample(pts, 1.0)
        x, y, a, _ = smp[len(smp) // 2]
        t = n.upper()
        tw = dr.textlength(t, font=F_SM) + 8
        tmp = Image.new("RGBA", (int(tw), 22), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((4, 2), t, font=F_SM, fill=INK)
        deg = -math.degrees(a)
        deg = deg - 180 if deg > 90 else deg + 180 if deg < -90 else deg
        tmp = tmp.rotate(deg, expand=True, resample=Image.BICUBIC)
        img.paste(tmp, (int(x * S - tmp.width / 2), int(y * S - tmp.height / 2)), tmp)
for ref, t, sz in [((1150, 180), "S U G I    F O R E S T", F_BIG), ((690, 715), "F U R U M A C H I", F_BIG),
                   ((690, 745), "(old quarter)", F_IT), ((268, 700), "KUZU RIVER", F_BIG), ((143, 615), "canal", F_IT),
                   ((150, 1010), "To Kamiyama (6 km)", F_IT), ((190, 1370), "<- to Terminal City", F_IT),
                   ((1760, 1380), "to Nagiri Jct. (22 km) ->", F_IT), ((1750, 1250), "FARMLAND", F_BIG),
                   ((400, 1470), "RICE PADDIES", F_BIG), ((1550, 1470), "RICE PADDIES", F_BIG), ((1180, 740), "vegetable plots", F_IT),
                   ((440, 1010), "jizo", F_SM), ((1345, 1000), "bus stop", F_SM), ((880, 815), "well", F_SM)]:
    text(*M(ref), t, sz)
# başlık kartuşu, ölçek, kuzey
tx0, ty0 = M((110, 110))
tx1, ty1 = M((730, 280))
dr.rectangle([P(tx0, ty0), P(tx1, ty1)], fill=PAPER, outline=INK, width=4)
dr.rectangle([P(tx0 + 1.2, ty0 + 1.2), P(tx1 - 1.2, ty1 - 1.2)], outline=INK, width=2)
dr.text(P(tx0 + 8, ty0 + 5), "MIZUNAGI-CHŌ", font=F_TITLE, fill=INK)
dr.text(P(tx0 + 10, ty0 + 23), "Postal Delivery Area · 1991", font=F_IT, fill=INK)
dr.text(P(tx0 + 10, ty0 + 30), f"plan {W:.0f} x {H:.0f} m · 1 m = {S:.0f} px · {len(buildings)} buildings", font=F_SM, fill=INK)
sx, sy = M((130, 1490))
for i in range(3):
    dr.rectangle([P(sx + i * 20, sy), P(sx + i * 20 + 20, sy + 1.6)], fill=INK if i % 2 == 0 else PAPER, outline=INK, width=2)
for i in range(4):
    text(sx + i * 20, sy - 3, f"{i * 20}", F_SM)
text(sx + 66, sy - 3, "m", F_SM)
nx_, ny_ = M((180, 400))
dr.rounded_rectangle([P(nx_ - 12, ny_ - 15), P(nx_ + 12, ny_ + 33)], radius=12, fill=PAPER, outline=INK, width=3)
text(nx_, ny_ - 9, "N", F_BIG)
dr.line([P(nx_, ny_ - 3), P(nx_, ny_ + 27)], fill=INK, width=3)
dr.line([P(nx_ - 4, ny_ + 2), P(nx_, ny_ - 3), P(nx_ + 4, ny_ + 2)], fill=INK, width=3)
dr.line([P(nx_ - 4, ny_ + 22), P(nx_ + 4, ny_ + 22)], fill=INK, width=3)
dr.rectangle([0, 0, img.width - 1, img.height - 1], outline=INK, width=8)

img.save(os.path.join(OUT, "mizunagi_plan.png"))
json.dump(dict(scale_m_per_ref_px=K, size_m=[W, round(H, 2)], render_px_per_m=S, qa=report, buildings=buildings,
               roads=[dict(name=n, kind=k, width_m=w, points=[[round(x, 2), round(y, 2)] for x, y in pts]) for n, k, w, pts in ROADS],
               areas=areas, river=dict(width_m=RIVER_W, points=[[round(x, 2), round(y, 2)] for x, y in RIVER]),
               canal=[[round(x, 2), round(y, 2)] for x, y in CANAL], rail_y=RAIL_Y,
               fields=[[[round(x, 2), round(y, 2)] for x, y in f] for f in FIELDS],
               lamps=lamps, utility_poles=poles, vending=vending, jizo=jizo, bus_stop=BUS_STOP, trees=len(trees)),
          open(os.path.join(OUT, "mizunagi_plan.json"), "w"), ensure_ascii=False, indent=1)
print("tamam ->", OUT)
