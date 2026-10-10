#!/usr/bin/env python3
"""Mizunagi-machi kasaba planı üreticisi (v3: parsel tabanlı, yoğunluk kademeli infografik).

1991, Nagano / Kiso esinli küçük dağ kasabası. Tamamen üstten görünüş, elde çizilmiş
mimari yerleşim planı: siyah mürekkep, krem kâğıt, soluk mavi su.

Yol ağı referans çizimden (2000x1600 px, çerçeve 70..1930) alınır ve metreye çevrilir:
1 ref px = 0.2366 m  ->  harita 440 x 345 m.

Bölgeler ve yoğunluk (merkezden dışa azalır):
  Furumachi   en yoğun; düzensiz dar parseller, eski ahşap evler, kura, birkaç boş ev
  Old Kaido   dar cepheli machiya'lar yola dizili, açık drenaj kanalı
  Shotengai   istasyon ile Old Kaido arasında kısa, bitişik dükkân sırası
  Station     postane, bisiklet garajı, meydan, otobüs durağı, birkaç işletme
  Core        orta yoğunluk, blok duvarlı evler
  East Road   yeni, seyrek; geri çekilmiş evler, ön bahçe, carport
  Farm belt   çeltik + sulama; ana ev-kura-ambar-sundurma ortak avlu kümeleri
  Mountain    3-4 izole ev, ormancı kulübesi, dinlenme kulübesi, radyo direği, tünel

Çalıştır: python3 generate.py  ->  out/mizunagi_plan.png, out/mizunagi_plan.json
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


def U(r):
    return rnd.uniform(*r)


def choose(pairs):
    r, acc = rnd.random(), 0.0
    for k, p in pairs:
        acc += p
        if r <= acc:
            return k
    return pairs[-1][0]


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


road, water, solid, reserved, taken, settle, fieldz, forestz, plant, clear = (grid() for _ in range(10))


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


def poly_mask(poly):
    im = Image.new("1", (GW, GH), 0)
    ImageDraw.Draw(im).polygon([(x * RES, y * RES) for x, y in poly], fill=1)
    return np.array(im, bool)


def stamp_poly(mask, poly, val=True):
    a = poly_mask(poly)
    if val:
        mask |= a
    else:
        mask &= ~a


def stamp_rect(mask, cx, cy, w, d, a, val=True):
    stamp_poly(mask, rect_corners(cx, cy, w, d, a), val)


# ------------------------------------------------------------ yükselti (kontur)
yy, xx = np.mgrid[0:GH, 0:GW] / RES
noise = np.zeros_like(xx)
for f, amp in ((0.021, 1.6), (0.047, 0.8), (0.11, 0.3)):
    ph = rnd.random() * 6.28
    noise += amp * np.sin(xx * f + ph) * np.cos(yy * f * 1.3 + ph * 0.7)
HEIGHT = (np.clip((140 - yy) / 120, 0, 1) ** 1.15 * 30
          + np.clip((xx - 300) / 140, 0, 1) * np.clip((250 - yy) / 200, 0, 1) * 14
          + np.clip((60 - xx) / 60, 0, 1) * np.clip((150 - yy) / 150, 0, 1) * 8
          + noise * np.clip((190 - yy) / 120, 0.15, 1))

# ------------------------------------------------------------ su: nehir, kanallar, sulama
RIVER = smooth(MP([(395, 60), (420, 220), (470, 370), (468, 480), (410, 620), (345, 760), (330, 880), (378, 965),
                   (322, 1060), (310, 1150), (360, 1270), (400, 1380), (402, 1545)]))
RIVER_W = 14.0
stamp_line(water, RIVER, RIVER_W)
stamp_line(solid, RIVER, RIVER_W + 9)                    # set + set yolu payı
# sulama: nehirden savak (weir) ile alınır, kanaldan tarlalara, mansapta tekrar nehre
CHANNELS = [
    ("West canal", 2.4, [(408, 612), (300, 612), (215, 606), (150, 600), (140, 760), (136, 1000), (132, 1250), (135, 1392), (138, 1410)]),
    ("South channel W", 1.6, [(395, 1410), (300, 1410), (138, 1410), (74, 1410)]),
    ("South channel E", 1.6, [(410, 1412), (700, 1412), (1004, 1412), (1300, 1412), (1560, 1412), (1926, 1412)]),
    ("Mountain stream", 1.6, [(1900, 880), (1890, 1000), (1880, 1100), (1870, 1175), (1700, 1176), (1575, 1178), (1575, 1300),
                              (1578, 1392), (1580, 1412)]),
]
CHANNELS = [(n, w, smooth(MP(p), 4)) for n, w, p in CHANNELS]
for n, w, pts in CHANNELS:
    stamp_line(water, pts, w)
    stamp_line(solid, pts, w + 1.0)
WEIRS = [M((408, 612)), M((395, 1410)), M((410, 1412))]
RAIL_Y = M((0, 1392))[1]
stamp_line(road, [(0, RAIL_Y), (W, RAIL_Y)], 6.0)

# ------------------------------------------------------------ yollar (ref px)
# tür: main 7 m, road 6-6.5 m, lane 4-5 m, alley (roji) 2.7 m, dirt (tarla/orman) 3-3.6 m, trail 3 m
ROADS_REF = [
    ("Old Kaido (west)", "road", 6.0, [(60, 975), (200, 975), (325, 972)]),
    ("Old Kaido", "road", 6.5, [(432, 968), (600, 966), (800, 964), (990, 962)]),
    ("Chuo-dori", "main", 7.0, [(1005, 958), (1006, 1150), (1004, 1392), (1004, 1545)]),
    ("Sando", "lane", 5.0, [(990, 962), (990, 800), (978, 680), (965, 625), (950, 600), (946, 566)]),
    ("Mountain Road", "road", 6.0, [(1010, 965), (1100, 890), (1190, 800), (1330, 715), (1430, 688), (1510, 630), (1570, 588),
                                     (1650, 520), (1750, 492), (1850, 508), (1898, 470), (1895, 400), (1860, 330), (1858, 275),
                                     (1890, 240), (1840, 200)]),
    ("East Road", "road", 6.5, [(1006, 1086), (1150, 1084), (1300, 1080), (1450, 1074), (1600, 1064), (1655, 1068),
                                 (1690, 1110), (1722, 1160), (1750, 1170)]),
    ("Sakamichi", "lane", 4.5, [(1750, 1170), (1740, 1060), (1708, 1012), (1745, 960), (1800, 900), (1845, 848)]),
    ("Town Hall St", "lane", 4.5, [(1006, 978), (1100, 974), (1215, 980)]),
    ("Upper Lane", "lane", 4.0, [(1060, 965), (1130, 905), (1250, 900), (1330, 905)]),
    ("New Lane", "lane", 4.0, [(1450, 1074), (1452, 1130), (1470, 1200)]),
    ("Market St", "lane", 4.5, [(1006, 1190), (1120, 1188), (1240, 1182)]),
    ("River Road", "road", 6.0, [(455, 968), (452, 1080), (458, 1160), (485, 1240), (540, 1280), (640, 1300), (800, 1300),
                                  (1004, 1300)]),
    ("Station Front", "road", 6.0, [(1004, 1300), (1150, 1300), (1300, 1300), (1430, 1300), (1560, 1300)]),
    ("Shotengai", "lane", 5.0, [(908, 966), (907, 1100), (906, 1200), (907, 1300)]),
    ("Roji A", "alley", 2.7, [(745, 966), (742, 1100), (740, 1200), (738, 1300)]),
    ("Roji C", "alley", 2.7, [(560, 966), (562, 1114), (575, 1200), (600, 1288)]),
    ("Roji D", "alley", 2.7, [(458, 1118), (562, 1114), (742, 1108)]),
    ("Roji E", "alley", 2.7, [(742, 1200), (820, 1200), (906, 1200)]),
    ("Roji G", "alley", 2.7, [(1100, 1086), (1100, 1150), (1098, 1220), (1100, 1300)]),
    ("Roji H", "alley", 2.7, [(1220, 1086), (1218, 1182), (1220, 1300)]),
    # Furumachi
    ("Furumachi D", "lane", 4.0, [(598, 966), (596, 880), (596, 800), (600, 700)]),
    ("Furumachi A", "lane", 4.0, [(596, 800), (700, 796), (792, 786)]),
    ("Furumachi B", "lane", 4.0, [(805, 690), (800, 790), (778, 880), (768, 964)]),
    ("Furumachi C", "lane", 4.0, [(800, 792), (900, 784), (985, 780)]),
    ("Well alley", "alley", 2.7, [(812, 825), (880, 830), (945, 838)]),
    ("Furumachi E", "alley", 2.7, [(470, 966), (480, 860), (520, 760), (600, 700)]),
    ("Furumachi F", "alley", 2.7, [(596, 880), (690, 878), (772, 876)]),
    # tarla / orman yolları
    ("Furumachi path", "dirt", 3.0, [(520, 760), (590, 662), (700, 655), (830, 645), (950, 625)]),
    ("Forest edge path", "dirt", 3.0, [(978, 640), (1150, 615), (1300, 595), (1385, 610)]),
    ("West Bank Lane", "dirt", 3.6, [(170, 600), (168, 800), (168, 975), (165, 1100), (168, 1300), (172, 1392), (175, 1525)]),
    ("SE farm road", "dirt", 3.6, [(1560, 1300), (1600, 1302), (1700, 1310), (1800, 1322), (1925, 1325)]),
    ("Farm road E", "dirt", 3.6, [(1650, 1150), (1610, 1220), (1600, 1302), (1602, 1392), (1605, 1525)]),
    ("Farm road S", "dirt", 3.6, [(1004, 1520), (1300, 1520), (1600, 1520), (1925, 1520)]),
    ("Farm road SW", "dirt", 3.6, [(175, 1520), (440, 1520), (700, 1520), (1004, 1520)]),
    ("Forestry road", "dirt", 3.0, [(1430, 688), (1385, 610), (1300, 540), (1180, 480), (1080, 445), (1030, 440)]),
    ("Forest link", "dirt", 3.0, [(1510, 630), (1560, 560), (1620, 470), (1690, 370)]),
    ("Levee path E", "trail", 3.0, [(372, 640), (395, 700), (370, 800), (400, 900), (440, 965)]),
    ("Levee path S", "trail", 3.0, [(440, 975), (415, 1080), (385, 1170), (420, 1270), (450, 1385)]),
    ("Bike trail", "trail", 3.0, [(1240, 1182), (1320, 1170), (1400, 1205), (1500, 1225), (1560, 1300)]),
    ("Old postal path", "post", 2.0, [(1100, 352), (1180, 380), (1250, 425), (1350, 390), (1450, 378), (1530, 342),
                                      (1620, 330), (1700, 312), (1790, 345), (1830, 250), (1820, 205)]),
]
ROADS = [(n, k, w, smooth(MP(p))) for n, k, w, p in ROADS_REF]
ROADS.append(("Kaido bridge", "bridge", 6.0, MP([(318, 972), (440, 968)])))
for n, k, w, pts in ROADS:
    stamp_line(road, pts, w + 0.6)
WIDTH_NOTES = {"Chuo-dori": .32, "Old Kaido": .55, "Shotengai": .4, "East Road": .45, "Mountain Road": .22, "River Road": .2,
               "Station Front": .7, "Furumachi A": .5, "Roji D": .3, "West Bank Lane": .3, "Forestry road": .5, "Levee path S": .5,
               "Farm road S": .4, "Old postal path": .45}

# ------------------------------------------------------------ bölgeler
TOWN = MP([(405, 690), (600, 662), (965, 642), (1020, 700), (1042, 940), (1180, 830), (1330, 760), (1345, 880), (1560, 905),
           (1700, 940), (1790, 1000), (1800, 1170), (1640, 1170), (1640, 1290), (1450, 1385), (400, 1385), (405, 1250),
           (445, 1000), (430, 900)])
stamp_poly(settle, TOWN)
FIELDS = [("paddy", MP([(72, 600), (128, 600), (125, 1395), (72, 1395)])),
          ("paddy", MP([(176, 620), (285, 620), (290, 1385), (176, 1385)])),
          ("paddy", MP([(72, 1416), (1928, 1416), (1928, 1528), (72, 1528)])),
          ("paddy", MP([(1585, 1182), (1928, 1182), (1928, 1385), (1585, 1385)])),
          ("veg", MP([(1060, 700), (1310, 690), (1420, 700), (1280, 760), (1150, 840), (1060, 870)]))]
for _, f in FIELDS:
    stamp_poly(fieldz, f)
fieldz &= ~road & ~water
settle &= ~fieldz
forestz[:] = ~settle & ~fieldz & ~water
stamp_poly(forestz, MP([(0, 1395), (1930, 1395), (1930, 1600), (0, 1600)]), False)
PLANTATIONS = [(MP(p), math.radians(a)) for p, a in [
    ([(1060, 90), (1480, 90), (1500, 250), (1300, 300), (1080, 280)], 8),
    ([(1560, 560), (1780, 520), (1880, 620), (1840, 760), (1680, 820), (1600, 720)], -20),
    ([(470, 300), (720, 300), (740, 520), (560, 600), (480, 560)], 0),
    ([(80, 300), (300, 300), (330, 560), (90, 560)], 12)]]
CLEARCUTS = [MP(p) for p in [[(1250, 140), (1360, 150), (1350, 240), (1240, 230)],
                             [(1640, 600), (1740, 590), (1750, 680), (1650, 690)],
                             [(560, 380), (650, 380), (650, 460), (560, 460)]]]
for p, _ in PLANTATIONS:
    stamp_poly(plant, p)
for p in CLEARCUTS:
    stamp_poly(clear, p)
plant &= forestz
clear &= plant

# ------------------------------------------------------------ yapılar
buildings, areas, parcels, extras = [], [], [], []


def free(cx, cy, w, d, a, margin):
    xs, ys = rect_samples(cx, cy, w + 2 * margin, d + 2 * margin, a)
    return not (any_hit(road, xs, ys) or any_hit(solid, xs, ys) or any_hit(water, xs, ys))


def add_building(typ, cx, cy, w, d, a, label=None, margin=0.5, nudge=False, **kw):
    cand = [(0, 0)] + ([(r * math.cos(t * math.pi / 4), r * math.sin(t * math.pi / 4))
                        for r in (1.5, 3, 4.5, 6) for t in range(8)] if nudge else [])
    for dx, dy in cand:
        if free(cx + dx, cy + dy, w, d, a, margin):
            b = dict(id=len(buildings), type=typ, x=round(cx + dx, 2), y=round(cy + dy, 2), w=round(w, 2), d=round(d, 2),
                     ang=round(a, 4), label=label, delivery=None, enterable=False)
            b.update(kw)
            buildings.append(b)
            stamp_rect(solid, b["x"], b["y"], w + 0.3, d + 0.3, a)
            return b
    if label:
        print("UYARI: yerleşmedi:", typ, label)
    return None


def special(typ, ref, wpx, hpx, label=None, deg=0.0, nudge=True, margin=0.6, **kw):
    cx, cy = M(ref)
    return add_building(typ, cx, cy, wpx * K, hpx * K, math.radians(deg), label, margin, nudge, **kw)


def area(typ, x0, y0, x1, y1, label=None, block=True):
    a, b = M((x0, y0)), M((x1, y1))
    areas.append(dict(type=typ, rect=[round(a[0], 2), round(a[1], 2), round(b[0], 2), round(b[1], 2)], label=label))
    quad = [a, (b[0], a[1]), b, (a[0], b[1])]
    stamp_poly(reserved, quad)
    if block:
        stamp_poly(solid, quad)


# --- Şinto mabedi (doğal teras): torii, taş basamak, temizuya, haiden, honden, shamusho, festival deposu, koruluk
SHRINE = dict(honden=M((946, 300)), grove_r=24.0)
special("honden", (946, 300), 24, 24, "Honden", nudge=False, margin=0.3)
special("haiden", (946, 352), 50, 34, "Haiden", nudge=False, margin=0.3)
special("temizuya", (980, 390), 14, 10, None, nudge=False, margin=0.2)
special("shamusho", (998, 352), 30, 24, "Shamusho", nudge=False, margin=0.3)
special("festival_store", (896, 352), 26, 18, "Festival store", nudge=False, margin=0.3)
special("kannushi_house", (1062, 338), 46, 34, "Kannushi house", nudge=False)
special("torii", (946, 562), 30, 5, None, nudge=False, margin=0)
special("torii", (946, 404), 26, 4, None, nudge=False, margin=0)
area("keidai", 878, 282, 1018, 400, None, block=False)
area("steps", 939, 408, 953, 556, None)
# --- istasyon önü: postane (shotengai'nin bitiş noktası), bisiklet garajı, meydan, durak, birkaç işletme
special("station", (1090, 1362), 100, 26, "Mizunagi Station", nudge=False, margin=0.4)
special("post_office", (908, 1343), 96, 48, "Post Office", nudge=False, margin=0.4, note="our room upstairs")
special("bike_garage", (825, 1347), 44, 34, "Bike garage", margin=0.4)
special("shokudo", (765, 1346), 38, 30, "Shokudo", margin=0.4)
special("kissaten", (1185, 1338), 34, 28, "Kissaten", margin=0.4)
special("ja_coop", (1250, 1340), 80, 48, "JA co-op", margin=0.4)
special("freight_shed", (1350, 1358), 70, 24, "Freight shed", margin=0.3)
area("plaza", 1024, 1314, 1160, 1346, "Station plaza")
area("platform", 1030, 1404, 1190, 1414, None)
BUS_STOP = M((1150, 1322))
# --- kamu / servis
special("police_box", (955, 1000), 24, 20, "Koban", margin=0.4)
special("sento", (690, 1060), 56, 40, "Sento", margin=0.5)
area("park", 585, 1000, 640, 1050, "Playground")
special("town_hall", (1135, 1030), 78, 40, "Town Office")
special("fire_station", (1290, 1040), 46, 32, "Fire station")
special("clinic", (1595, 1150), 66, 44, "Clinic")
area("parking", 1630, 1190, 1680, 1230, "P")
special("market", (1395, 1262), 90, 44, "Market", margin=0.4)
area("parking", 1452, 1240, 1500, 1288, "P")
special("school", (1440, 935), 170, 40, "Mizunagi Primary School", nudge=False)
special("school_wing", (1540, 905), 44, 82, None, nudge=False)
special("gym", (1328, 955), 52, 38, "Gym")
area("schoolyard", 1360, 965, 1500, 1028, None)
special("community_hall", (1600, 1000), 72, 38, "Community hall")
special("water_tower", (1868, 818), 26, 26, "Water tower", margin=0.2)
# --- dağ bölgesi
special("forester_hut", (1495, 672), 36, 30, "Forester hut", deg=-35)
special("lookout", (1715, 592), 30, 14, "Lookout rock", margin=0.2)
special("radio_mast", (1525, 270), 18, 18, "Radio mast", margin=0.2)
special("rest_hut", (1790, 352), 32, 22, "Old postmen's rest hut")
special("tunnel", (1820, 185), 34, 24, None, margin=0)
for i, p in enumerate([(1478, 328), (1555, 405), (1395, 405), (1640, 360)]):
    b = special("mountain_house", p, 40, 32, "Mountain house" if i == 0 else None, deg=rnd.uniform(-18, 18), zone="mountain")
    if b:
        parcels.append(dict(id=len(parcels), zone="mountain", wall="hedge", poly=rect_corners(b["x"], b["y"], b["w"] + 7, b["d"] + 7, b["ang"]),
                            front=[], gate=None, built=True))
        stamp_rect(taken, b["x"], b["y"], b["w"] + 7, b["d"] + 7, b["ang"])


# --- çiftlik kümeleri: ana ev + kura + ambar + sundurma ortak avlu (doma niwa) çevresinde, kuzey/batıda rüzgârlık
def farm_cluster(ref, deg=0.0):
    cx, cy = M(ref)
    a = math.radians(deg)
    found = None
    for sc in (1.0, 0.9, 0.8):
        w_, d_ = 25.0 * sc, 22.0 * sc
        for r in (0, 2, 4, 6, 8, 10, 12):
            for t in range(8 if r else 1):
                nx, ny = cx + r * math.cos(t * math.pi / 4), cy + r * math.sin(t * math.pi / 4)
                if free(nx, ny, w_, d_, a, 0.5) and not any_hit(taken | reserved, *rect_samples(nx, ny, w_, d_, a, 1.0)):
                    found = (nx, ny, sc)
                    break
            if found:
                break
        if found:
            break
    if not found:
        print("UYARI: çiftlik yerleşmedi", ref)
        return
    cx, cy, sc = found
    c, s_ = math.cos(a), math.sin(a)

    def L(u, v):
        return cx + u * c - v * s_, cy + u * s_ + v * c
    pid = len(parcels)
    parcels.append(dict(id=pid, zone="farm", wall="hedge", poly=rect_corners(cx, cy, w_, d_, a), front=[], gate=None, built=True))
    stamp_rect(taken, cx, cy, w_, d_, a)
    stamp_rect(solid, cx, cy, w_ + 0.6, d_ + 0.6, a)
    parts = [("farmhouse", 0.0, -6.0, 12.0, 8.5), ("kura", 8.4, 1.0, 4.5, 5.5), ("barn", -7.8, 2.0, 7.0, 6.0), ("open_shed", 0.5, 8.0, 7.0, 3.6)]
    for typ, u, v, bw, bd in [(t_, u_ * sc, v_ * sc, bw_ * sc, bd_ * sc) for t_, u_, v_, bw_, bd_ in parts]:
        x, y = L(u, v)
        b = dict(id=len(buildings), type=typ, x=round(x, 2), y=round(y, 2), w=bw, d=bd, ang=round(a, 4), label=None,
                 delivery=None, enterable=False, zone="farm", parcel=pid)
        if typ == "farmhouse":
            gx, gy = L(0.0, -1.0)
            b.update(genkan=[round(gx, 2), round(gy, 2)], mailbox=[round(L(-10.5, -d_ / 2 + 0.6)[0], 2), round(L(-10.5, -d_ / 2 + 0.6)[1], 2)])
        buildings.append(b)
    extras.append(("courtyard", rect_corners(*L(0.5 * sc, 0.8 * sc), 9.5 * sc, 6.0 * sc, a)))
    for k in range(9):                    # rüzgârlık ağaçları (yashikirin)
        extras.append(("windbreak", L(-w_ / 2 + 1.2, -d_ / 2 + 1.2 + k * 2.4 * sc)))
        extras.append(("windbreak", L(-w_ / 2 + 1.2 + k * 2.6 * sc, -d_ / 2 + 1.0)))


for ref, deg in [((230, 850), 0), ((230, 1225), 0), ((1715, 1240), 3), ((1850, 1250), -4), ((560, 1466), 0), ((1700, 1466), 0)]:
    farm_cluster(ref, deg)

# ------------------------------------------------------------ parsel tabanlı konut yerleşimi
ZP = {
    "shop":      dict(pw=(5.0, 7.0), pd=(12, 14), sb=(0.0, 0.1), hd=(9, 11), side=(0.0, 0.1), p=0.97, cap=14,
                      walls=[("none", 1.0)], kura=.1, shed=.25, car=0, ditch=False, passage=.08, kinds=[("shop", 1.0)]),
    "kaido":     dict(pw=(5.0, 7.0), pd=(12, 16), sb=(0.0, 0.3), hd=(9, 12), side=(0.0, 0.25), p=0.9, cap=22,
                      walls=[("none", .55), ("fence", .45)], kura=.45, shed=.15, car=0, ditch=True, passage=.3,
                      kinds=[("machiya", .82), ("akiya", .1), ("old_house", .08)]),
    "furumachi": dict(pw=(5.5, 8.0), pd=(9.5, 13), sb=(0.2, 0.7), hd=(6.5, 8.5), side=(0.15, 0.5), p=0.95, cap=42,
                      walls=[("fence", .45), ("stone", .3), ("none", .25)], kura=.35, shed=.25, car=0, ditch=True, passage=.4,
                      kinds=[("old_house", .74), ("machiya", .12), ("akiya", .14)]),
    "core":      dict(pw=(8.0, 10.5), pd=(11, 14), sb=(0.8, 1.8), hd=(7.0, 9.0), side=(0.8, 1.6), p=0.85, cap=42,
                      walls=[("block", .45), ("fence", .25), ("hedge", .15), ("none", .15)], kura=.05, shed=.4, car=.25,
                      ditch=False, passage=.35, kinds=[("house", .8), ("old_house", .12), ("akiya", .08)]),
    "east":      dict(pw=(11.5, 14.5), pd=(12, 15), sb=(2.6, 3.8), hd=(7.5, 9.0), side=(1.6, 3.0), p=0.8, cap=26,
                      walls=[("hedge", .45), ("block", .4), ("none", .15)], kura=0, shed=.3, car=.8, ditch=False, passage=.15,
                      kinds=[("house", .9), ("apartment", .1)]),
    "mtnroad":   dict(pw=(14, 18), pd=(13, 17), sb=(3, 5), hd=(8, 10), side=(3, 6), p=0.5, cap=14,
                      walls=[("hedge", .5), ("none", .5)], kura=.25, shed=.5, car=.5, ditch=False, passage=0,
                      kinds=[("house", .55), ("old_house", .45)]),
    "rural":     dict(pw=(16, 20), pd=(15, 19), sb=(4, 6), hd=(9.5, 12), side=(3, 6), p=0.6, cap=8,
                      walls=[("hedge", .7), ("none", .3)], kura=.5, shed=.6, car=.3, ditch=True, passage=0,
                      kinds=[("farmhouse", 1.0)]),
}
ROAD_ZONE = {"Shotengai": "shop", "Old Kaido": "kaido", "Old Kaido (west)": "kaido",
             "Furumachi A": "furumachi", "Furumachi B": "furumachi", "Furumachi C": "furumachi", "Furumachi D": "furumachi",
             "Furumachi E": "furumachi", "Furumachi F": "furumachi", "Well alley": "furumachi",
             "Chuo-dori": "core", "River Road": "core", "Roji A": "core", "Roji C": "core", "Roji D": "core", "Roji E": "core",
             "Station Front": "core", "Market St": "east", "Roji G": "core", "Roji H": "east", "Furumachi path": "furumachi",
             "East Road": "east", "Town Hall St": "east", "Upper Lane": "east", "Sakamichi": "mtnroad",
             "Mountain Road": "mtnroad", "New Lane": "east", "West Bank Lane": "rural", "SE farm road": "rural"}
ZONE_ORDER = ["shop", "kaido", "furumachi", "core", "east", "mtnroad", "rural"]
DITCH_ROADS = [n for n, z in ROAD_ZONE.items() if ZP[z]["ditch"]]
count = {z: 0 for z in ZP}
passages = []


def furnish(P_, Wp, pw, pd, zone, name, zp, n, a, gaps, gate_at=None, mail_at=None, no_car=False):
    """Parselin içini kur: ev, genkan, kapı, posta kutusu, carport, kura/depo, bahçe."""
    pid = P_["id"]
    if rnd.random() < zp["p"] and count[zone] < zp["cap"]:
        sg = U(zp["side"])
        car = rnd.random() < zp["car"] and not no_car
        cs = rnd.choice((-1, 1))
        hw = max(4.2, min(13.0, pw - 2 * sg - (3.6 if car else 0)))
        sb = U(zp["sb"])
        hd = min(U(zp["hd"]), pd - sb - 1.2)
        uc = -cs * 1.8 if car else U((-(pw - 2 * sg - hw) / 2, (pw - 2 * sg - hw) / 2)) if pw - 2 * sg > hw else 0.0
        typ = choose(zp["kinds"])
        hx, hy = Wp(uc, sb + hd / 2)
        b = add_building(typ, hx, hy, hw, hd, a, None, 0.05, zone=zone, parcel=pid, road=name)
        if b:
            count[zone] += 1
            P_["built"] = True
            ug = max(-pw / 2 + 1.4, min(pw / 2 - 1.4, uc + U((-hw / 4, hw / 4)))) if gate_at is None else gate_at
            gu = max(uc - hw / 2 + 0.7, min(uc + hw / 2 - 0.7, ug))
            gx, gy = Wp(gu, sb)
            b["genkan"] = [round(gx, 2), round(gy, 2)]
            b["genkan_dir"] = round(math.atan2(-n[1], -n[0]), 3)
            if sb >= 1.4:
                px_, py_ = Wp(gu, sb - 0.55)
                add_building("porch", px_, py_, 1.8, 1.0, a, None, 0.0, zone=zone, parcel=pid)
            mside = 1 if ug < pw / 2 - 2.8 else -1
            mx, my = Wp(ug + mside * 1.6, 0.35)
            if mail_at is not None:
                mx, my = mail_at
            b["mailbox"] = [round(mx, 2), round(my, 2)]
            gaps.append((ug - 0.6, ug + 0.6) if gate_at is not None else (ug - 1.1, ug + 1.1))
            P_["gate"] = [Wp(ug - 0.6 if gate_at is not None else ug - 1.1, 0), Wp(ug + 0.6 if gate_at is not None else ug + 1.1, 0)]
            back0 = sb + hd + 0.8
            if car:
                ccx = cs * (pw / 2 - sg - 1.6)
                dep = min(5.5, max(4.8, sb + hd * 0.6))
                if zone in ("east", "mtnroad", "rural"):
                    cx_, cy_ = Wp(ccx, 0.4 + dep / 2)
                    add_building("carport", cx_, cy_, 3.0, dep, a, None, 0.05, zone=zone, parcel=pid)
                else:
                    cx_, cy_ = Wp(ccx, 0.3 + 2.5)
                    extras.append(("parking_pocket", rect_corners(cx_, cy_, 2.6, 4.8, a)))
                gaps.append((ccx - 1.6, ccx + 1.6))
            if rnd.random() < zp["kura"] and pd - back0 >= 5.5:
                kx, ky = Wp(U((-pw / 2 + 2.6, pw / 2 - 2.6)) if pw > 6 else 0, back0 + 2.4)
                add_building("kura", kx, ky, min(4.0, pw - 1.2), 4.6, a, None, 0.1, zone=zone, parcel=pid)
            elif rnd.random() < zp["shed"] and pd - back0 >= 3.2:
                kx, ky = Wp((pw / 2 - 1.8) * rnd.choice((-1, 1)), back0 + 1.4)
                add_building("shed", kx, ky, 2.6, 2.4, a, None, 0.05, zone=zone, parcel=pid)
            if pd - back0 >= 3.0 and rnd.random() < .55:
                extras.append(("veg_rows", [Wp(-pw / 2 + 0.8, pd - 0.6), Wp(pw / 2 - 0.8, pd - 0.6), Wp(pw / 2 - 0.8, back0 + 0.4),
                                            Wp(-pw / 2 + 0.8, back0 + 0.4)]))
            if sb > 2.5:
                for _ in range(rnd.randint(2, 4)):
                    u_ = U((-pw / 2 + 0.8, pw / 2 - 0.8))
                    if abs(u_ - ug) > 1.4 and not (car and abs(u_ - cs * (pw / 2 - sg - 1.6)) < 2.0):
                        extras.append(("shrub", Wp(u_, U((0.7, sb - 0.7)))))
                extras.append(("path", [Wp(ug, 0.2), Wp(ug, sb - 1.0)]))
    else:
        if zone not in ("shop",):
            extras.append(("veg_rows", [Wp(-pw / 2 + 0.8, pd - 0.8), Wp(pw / 2 - 0.8, pd - 0.8), Wp(pw / 2 - 0.8, 0.8),
                                        Wp(-pw / 2 + 0.8, 0.8)]))


def front_segments(Wp, pw, gaps):
    out, cur = [], -pw / 2
    for g0, g1 in sorted(gaps):
        if g0 > cur + 0.2:
            out.append([Wp(cur, 0), Wp(g0, 0)])
        cur = max(cur, g1)
    if cur < pw / 2 - 0.2:
        out.append([Wp(cur, 0), Wp(pw / 2, 0)])
    return out


def build_parcels(name, kind, wd, pts, zone):
    zp = ZP[zone]
    smp = resample(pts, 0.5)
    if len(smp) < 6:
        return
    L = smp[-1][3]
    allow = (settle | fieldz) if zone == "rural" else (settle | (forestz & ~plant)) if zone == "mtnroad" else settle
    for side in (-1, 1):
        s = U((0.5, 2.5))
        while s < L - 3:
            if count[zone] >= zp["cap"]:
                return
            pw, pd = U(zp["pw"]), U(zp["pd"])
            sc = s + pw / 2
            if sc > L - 1.5:
                break
            x, y, a, _ = smp[min(len(smp) - 1, int(sc / 0.5))]
            o0 = wd / 2 + 0.9 + (0.8 if zp["ditch"] else 0.0)
            t = (math.cos(a), math.sin(a))
            n = (-math.sin(a) * side, math.cos(a) * side)

            def Wp(u, v, x=x, y=y, t=t, n=n, o0=o0):
                return (x + t[0] * u + n[0] * (o0 + v), y + t[1] * u + n[1] * (o0 + v))
            pd_min = zp["sb"][0] + zp["hd"][0] + 1.4
            ok = False
            for f_ in (1.0, 0.85, 0.72, 0.6):              # blok sığ ise parseli kısalt
                pdt = max(pd_min, pd * f_)
                cx, cy = Wp(0, pdt / 2)
                xs, ys = rect_samples(cx, cy, pw - 0.5, pdt - 0.5, a, 0.6)
                if not (any_hit(road, xs, ys) or any_hit(solid, xs, ys) or any_hit(water, xs, ys) or any_hit(taken, xs, ys)
                        or any_hit(reserved, xs, ys) or not all_in(allow, xs, ys)):
                    ok = True
                    break
                if pdt <= pd_min:
                    break
            if not ok:
                s += 0.4
                continue
            pd = pdt
            stamp_rect(taken, cx, cy, pw, pd, a)
            pid = len(parcels)
            P_ = dict(id=pid, zone=zone, road=name, wall=choose(zp["walls"]), built=False,
                      poly=[Wp(-pw / 2, 0), Wp(pw / 2, 0), Wp(pw / 2, pd), Wp(-pw / 2, pd)], front=[], gate=None)
            parcels.append(P_)
            gaps = []
            gate_u = furnish(P_, Wp, pw, pd, zone, name, zp, n, a, gaps)
            # ön duvar parçaları (kapı ve araç açıklıkları hariç)
            P_["front"] = front_segments(Wp, pw, gaps)
            step = pw
            if rnd.random() < zp["passage"]:          # dar servis geçidi -> arka sıra bayrak parsel (hatazao-chi)
                pu = pw / 2 + 0.6
                pa = [Wp(pu, 0), Wp(pu, pd + 0.3)]
                step += 1.2
                pw2, pd2 = U(zp["pw"]) + 1.2, max(pd_min, U(zp["pd"]) * 0.85)
                u0 = pu - pw2 / 2 + 0.6

                def Wr(u, v, Wp=Wp, u0=u0, pd=pd):
                    return Wp(u0 + u, pd + 0.3 + v)
                rx, ry = Wr(0, pd2 / 2)
                xs, ys = rect_samples(rx, ry, pw2 - 0.5, pd2 - 0.5, a, 0.6)
                if (count[zone] < zp["cap"] and not (any_hit(road, xs, ys) or any_hit(solid, xs, ys) or any_hit(water, xs, ys)
                                                     or any_hit(taken, xs, ys) or any_hit(reserved, xs, ys)) and all_in(allow, xs, ys)):
                    stamp_rect(taken, rx, ry, pw2, pd2, a)
                    R_ = dict(id=len(parcels), zone=zone, road=name, wall=choose(zp["walls"]), built=False, flag_lot=True,
                              poly=[Wr(-pw2 / 2, 0), Wr(pw2 / 2, 0), Wr(pw2 / 2, pd2), Wr(-pw2 / 2, pd2)], front=[], gate=None)
                    parcels.append(R_)
                    g2 = []
                    furnish(R_, Wr, pw2, pd2, zone, name, zp, n, a, g2, gate_at=pu - u0, mail_at=Wp(pu, 0.35), no_car=True)
                    R_["front"] = front_segments(Wr, pw2, g2)
                    passages.append(pa)
                else:
                    passages.append([Wp(pu, 0), Wp(pu, pd * 0.9)])
            s += step


roads_by_zone = {z: [r for r in ROADS if ROAD_ZONE.get(r[0]) == z] for z in ZONE_ORDER}
for z in ZONE_ORDER:
    rs = roads_by_zone[z][:]
    rnd.shuffle(rs)
    for name, kind, wd, pts in rs:
        build_parcels(name, kind, wd, pts, z)

# ------------------------------------------------------------ teslimat adresleri ve girilebilir yapılar
RESID = {"old_house", "machiya", "house", "apartment", "akiya", "farmhouse", "mountain_house"}
houses = [b for b in buildings if b["type"] in RESID]
shops = [b for b in buildings if b["type"] == "shop"]
shop_on_st = sorted(shops, key=lambda b: b["y"])
for b, name in zip(shop_on_st[1::3], ["Ramen-ya", "Barber", "Bicycle shop", "Sake shop", "Tofu-ya"]):
    b["label"] = name


def pick_delivery(zone, k, types=RESID):
    cand = [b for b in buildings if b.get("zone") == zone and b["type"] in types and b["type"] != "akiya" and b["delivery"] is None]
    rnd.shuffle(cand)
    return cand[:k]


route = (pick_delivery("shop", 4, {"shop"}) + pick_delivery("kaido", 4) + pick_delivery("furumachi", 7) + pick_delivery("core", 6)
         + pick_delivery("east", 5) + pick_delivery("mtnroad", 1) + pick_delivery("farm", 2) + pick_delivery("mountain", 2)
         + [b for b in buildings if b["type"] in ("ja_coop", "clinic", "town_hall")])
for i, b in enumerate(route, 1):
    b["delivery"] = i
akiya_fm = [b for b in buildings if b["type"] == "akiya" and b.get("zone") == "furumachi"]
if akiya_fm:
    akiya_fm[0]["label"] = "Empty house"
ENTER = {"post_office", "station", "police_box", "clinic", "school", "sento", "ja_coop", "shamusho", "rest_hut"}
for b in buildings:
    if b["type"] in ENTER or b["label"] in ("Ramen-ya", "Empty house"):
        b["enterable"] = True

# ------------------------------------------------------------ donatılar
poles = []
for name, kind, wd, pts in ROADS:
    if kind in ("trail", "bridge", "post"):
        continue
    for nn, (x, y, a, _) in enumerate(resample(pts, 32)[1:]):
        side = 1 if nn % 2 else -1
        px_, py_ = x - math.sin(a) * side * (wd / 2 + 0.5), y + math.cos(a) * side * (wd / 2 + 0.5)
        if not (at(water, px_, py_) or at(solid, px_, py_)):
            poles.append((round(px_, 1), round(py_, 1)))
JIZO = [M((440, 990)), M((606, 706))]
LANTERNS = [M((930, 400)), M((962, 400)), M((932, 555)), M((960, 555))]


# ------------------------------------------------------------ QA
def qa():
    over_road = in_water = overlaps = parcel_road = 0
    for b in buildings:
        if b["type"] in ("tunnel", "torii"):
            continue
        xs, ys = rect_samples(b["x"], b["y"], b["w"] - .1, b["d"] - .1, b["ang"], 0.4)
        over_road += any_hit(road, xs, ys)
        in_water += any_hit(water, xs, ys)
    for i, a in enumerate(buildings):
        for b in buildings[i + 1:]:
            if abs(a["x"] - b["x"]) > 25 or abs(a["y"] - b["y"]) > 25:
                continue
            xs, ys = rect_samples(a["x"], a["y"], a["w"] - .2, a["d"] - .2, a["ang"], 0.5)
            c, s_ = math.cos(-b["ang"]), math.sin(-b["ang"])
            u = (xs - b["x"]) * c - (ys - b["y"]) * s_
            v = (xs - b["x"]) * s_ + (ys - b["y"]) * c
            overlaps += bool(np.any((abs(u) < b["w"] / 2 - .1) & (abs(v) < b["d"] / 2 - .1)))
    for p in parcels:
        xs_ = np.array([q[0] for q in p["poly"]])
        ys_ = np.array([q[1] for q in p["poly"]])
        cx, cy = xs_.mean(), ys_.mean()
        parcel_road += any_hit(road, np.array([(x + cx) / 2 for x in xs_]), np.array([(y + cy) / 2 for y in ys_]))
    by_zone = {}
    for b in buildings:
        if b["type"] in RESID:
            z = b.get("zone", "?")
            by_zone[z] = by_zone.get(z, 0) + 1
    types = {}
    for b in buildings:
        types[b["type"]] = types.get(b["type"], 0) + 1
    nonres = [b for b in buildings if b["label"] and b["type"] not in RESID] + shops
    return dict(dwellings=len([b for b in buildings if b["type"] in RESID]) + len(shops),
                houses=len([b for b in buildings if b["type"] in RESID]), shophouses=len(shops),
                named_nonresidential=len({b["id"] for b in nonres}),
                delivery_addresses=sum(1 for b in buildings if b["delivery"]),
                enterable=sum(1 for b in buildings if b["enterable"]),
                houses_by_zone=by_zone, parcels=len(parcels),
                buildings_on_road=int(over_road), buildings_in_water=int(in_water), building_overlaps=int(overlaps),
                parcels_on_road=int(parcel_road), by_type=types)


report = qa()
print(json.dumps(report, ensure_ascii=False, indent=1))

# ------------------------------------------------------------ çizim
PAPER, INK, GREY, LIGHT, WATER, WATER_D = (242, 237, 224), (30, 28, 25), (125, 120, 110), (176, 170, 158), (204, 222, 230), (110, 140, 155)
MAP_H = int(H * S)
LEG_H = 430
img = Image.new("RGB", (int(W * S), MAP_H + LEG_H), PAPER)


def P(x, y):
    return (x * S, y * S)


def wob(pts, amp=0.06):
    return [(x * S + rnd.uniform(-amp, amp) * S, y * S + rnd.uniform(-amp, amp) * S) for x, y in pts]


def font(names, size):
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            pass
    return ImageFont.load_default()


OBL = ["/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"]
BLD = ["/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
F_IT, F_SM, F_XS = font(OBL, 17), font(OBL, 14), font(OBL, 11)
F_BIG, F_TITLE = font(BLD, 22), font(BLD, 50)
F_NUM = font(BLD, 11)
F_JP = font(["/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"], 30)

# kontur (2 m) — yol, su, parsel üstüne çizilmez
cont = np.zeros((GH, GW), bool)
h2 = np.floor(HEIGHT / 2.0)
cont[:, 1:] |= h2[:, 1:] != h2[:, :-1]
cont[1:, :] |= h2[1:, :] != h2[:-1, :]
cont &= ~road & ~water & ~taken & ~reserved & ((~settle) | (HEIGHT > 3))
ci = Image.fromarray((cont * 255).astype(np.uint8)).resize((int(W * S), MAP_H), Image.NEAREST)
canvas = Image.new("L", img.size, 0)
canvas.paste(ci, (0, 0))
img.paste(Image.new("RGB", img.size, (200, 193, 177)), mask=canvas)
dr = ImageDraw.Draw(img)

# tarlalar: çeltik (su çizgili, sırt arası hendekli), bostan (sıra)
for kind_, f in FIELDS:
    x0, x1 = min(p[0] for p in f), max(p[0] for p in f)
    y0, y1 = min(p[1] for p in f), max(p[1] for p in f)
    cw, ch = ((8, 6) if kind_ == "veg" else rnd.choice([(16, 11), (15, 10), (18, 12)]))
    y, row = y0, 0
    while y < y1:
        x = x0 + rnd.uniform(-3, 0)
        while x < x1:
            w_ = cw + rnd.uniform(-2, 2)
            xs, ys = rect_samples(x + w_ / 2, y + ch / 2, w_ - 1, ch - 1, 0, 1.0)
            if all_in(fieldz, xs, ys) and not any_hit(solid | taken, xs, ys):
                dr.polygon(wob(rect_corners(x + w_ / 2, y + ch / 2, w_ - 0.8, ch - 0.8, 0), .08), outline=GREY, width=2)
                if kind_ == "paddy":
                    for k in range(1, 4):
                        dr.line([P(x + 1.5, y + k * ch / 4), P(x + w_ - 2.3, y + k * ch / 4)], fill=(170, 192, 200), width=1)
                else:
                    xx_ = x + 1.2
                    while xx_ < x + w_ - 1.2:
                        dr.line([P(xx_, y + 1.0), P(xx_, y + ch - 1.8)], fill=LIGHT, width=1)
                        xx_ += 1.0
            x += w_
        if kind_ == "paddy" and row % 2 == 1:             # tarla içi sulama hendeği
            yy_ = y + ch - 0.4
            seg = []
            for xx_ in np.arange(x0, x1, 0.5):
                ok = at(fieldz, xx_, yy_) and not at(solid | taken | road, xx_, yy_)
                if ok:
                    seg.append(P(xx_, yy_))
                elif len(seg) > 1:
                    dr.line(seg, fill=WATER_D, width=1)
                    seg = []
                else:
                    seg = []
            if len(seg) > 1:
                dr.line(seg, fill=WATER_D, width=1)
        y += ch
        row += 1

# alanlar
for a in areas:
    x0, y0, x1, y1 = a["rect"]
    t = a["type"]
    if t == "parking":
        dr.rectangle([P(x0, y0), P(x1, y1)], outline=INK, width=2)
        nst = max(1, int((x1 - x0) / 2.5))
        for k in range(1, nst):
            xx_ = x0 + k * (x1 - x0) / nst
            dr.line([P(xx_, y0), P(xx_, y0 + min(5, (y1 - y0) / 2.2))], fill=GREY, width=1)
            if y1 - y0 > 11:
                dr.line([P(xx_, y1 - 5), P(xx_, y1)], fill=GREY, width=1)
    elif t == "steps":                                   # taş basamak: 3 kol, 2 sahanlık
        L_ = y1 - y0
        for yy_ in np.arange(y0, y1, 0.6):
            f_ = (yy_ - y0) / L_
            if 0.31 < f_ < 0.38 or 0.64 < f_ < 0.71:
                continue
            dr.line([P(x0, yy_), P(x1, yy_)], fill=INK, width=1)
        dr.rectangle([P(x0, y0), P(x1, y1)], outline=INK, width=2)
    elif t == "schoolyard":
        dr.rounded_rectangle([P(x0 + 2, y0 + 2), P(x1 - 2, y1 - 2)], radius=int((y1 - y0) * S / 2.4), outline=INK, width=2)
        dr.rounded_rectangle([P(x0 + 5, y0 + 4), P(x1 - 5, y1 - 4)], radius=int((y1 - y0) * S / 3), outline=GREY, width=1)
    elif t == "plaza":
        dr.rectangle([P(x0, y0), P(x1, y1)], outline=INK, width=2)
        for xx_ in np.arange(x0 + 1, x1, 2.0):
            for yy_ in np.arange(y0 + 1, y1, 2.0):
                dr.point(P(xx_, yy_), fill=GREY)
        dr.ellipse([P((x0 + x1) / 2 - 2.2, (y0 + y1) / 2 - 2.2), P((x0 + x1) / 2 + 2.2, (y0 + y1) / 2 + 2.2)], outline=INK, width=2)
    elif t == "keidai":                                  # mabet avlusu: çakıl, taş duvar
        dr.rectangle([P(x0, y0), P(x1, y1)], outline=INK, width=4)
        for xx_ in np.arange(x0 + 1, x1, 1.6):
            for yy_ in np.arange(y0 + 1, y1, 1.6):
                if not at(solid, xx_, yy_):
                    dr.point(P(xx_, yy_), fill=LIGHT)
    elif t in ("park", "platform"):
        dr.rectangle([P(x0, y0), P(x1, y1)], outline=INK, width=1 if t == "park" else 2)
        if t == "park":
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            dr.ellipse([P(cx - 1.2, cy - 1.2), P(cx + 1.2, cy + 1.2)], outline=INK, width=2)
            dr.line([P(x0 + 2, y1 - 2), P(x0 + 6, y1 - 2)], fill=INK, width=2)

# orman: doğal karışık orman / sugi plantasyonu sıraları / kesim açıklığı / kutsal koruluk
blk = road | solid | water | taken | reserved
trees = []


def tri(x, y, h, w_, width=2):
    dr.polygon([P(x, y - h), P(x - w_, y), P(x + w_, y)], fill=PAPER, outline=INK, width=width)


def broadleaf(x, y, r=1.0):
    for dx, dy in ((-0.7 * r, 0.2 * r), (0.7 * r, 0.2 * r), (0, -0.6 * r)):
        dr.ellipse([P(x + dx - r, y + dy - r), P(x + dx + r, y + dy + r)], fill=PAPER, outline=INK, width=2)


gx_, gy_ = SHRINE["honden"]
for poly, ang in PLANTATIONS:                            # düzenli dikim sıraları
    pm = poly_mask(poly) & plant & ~clear
    xs_ = [p[0] for p in poly]
    ys_ = [p[1] for p in poly]
    cx, cy = sum(xs_) / len(xs_), sum(ys_) / len(ys_)
    R = max(max(xs_) - min(xs_), max(ys_) - min(ys_))
    c, s_ = math.cos(ang), math.sin(ang)
    for v in np.arange(-R, R, 3.4):
        for u in np.arange(-R, R, 2.6):
            x, y = cx + u * c - v * s_, cy + u * s_ + v * c
            if 0 < x < W and 0 < y < H and at(pm, x, y) and not any(at(blk, x + dx, y + dy) for dx, dy in ((0, 0), (0, -3.5), (0, .8))):
                trees.append(("plant", x, y))
for gy in np.arange(0, H, 5.0):                          # doğal orman: düzensiz, karışık
    for gx in np.arange(0, W, 5.0):
        x, y = gx + rnd.uniform(0, 5), gy + rnd.uniform(0, 5)
        if not at(forestz, x, y) or at(plant, x, y) or any(at(blk, x + dx, y + dy) for dx, dy in ((0, 0), (0, -5), (-2, 0), (2, 0), (0, 1))):
            continue
        near_grove = math.hypot(x - gx_, y - gy_) < SHRINE["grove_r"]
        trees.append(("grove" if near_grove else ("broad" if rnd.random() < .38 else "sugi"), x, y))
for p in CLEARCUTS:                                      # kesim açıklığı: kütükler ve fidanlar
    pm = poly_mask(p)
    for _ in range(140):
        x, y = rnd.uniform(min(q[0] for q in p), max(q[0] for q in p)), rnd.uniform(min(q[1] for q in p), max(q[1] for q in p))
        if at(pm, x, y) and not at(blk, x, y):
            trees.append(("stump" if rnd.random() < .7 else "sapling", x, y))
for _ in range(1500):                                    # kasaba içi bahçe ağaçları (seyrek)
    x, y = rnd.uniform(0, W), rnd.uniform(0, H)
    if not at(settle, x, y) or at(blk | fieldz, x, y):
        continue
    xs, ys = rect_samples(x, y, 3.6, 3.6, 0, 0.7)
    if any_hit(blk, xs, ys) or any((x - tx) ** 2 + (y - ty) ** 2 < 30 for t, tx, ty in trees[-300:] if t == "broad_t"):
        continue
    if rnd.random() < .45:
        trees.append(("broad_t", x, y))
for t, x, y in sorted(trees, key=lambda q: q[2]):
    if t == "plant":
        tri(x, y, 3.2, 1.15)
    elif t == "sugi":
        h_ = rnd.uniform(4.4, 5.8)
        tri(x + rnd.uniform(-.4, .4), y, h_, h_ * 0.36)
        dr.line([P(x - .8, y - h_ * .42), P(x + .8, y - h_ * .42)], fill=INK, width=1)
    elif t in ("broad", "broad_t"):
        broadleaf(x, y, rnd.uniform(0.9, 1.25))
    elif t == "grove":
        broadleaf(x, y, rnd.uniform(1.4, 1.8))
    elif t == "stump":
        dr.ellipse([P(x - .3, y - .3), P(x + .3, y + .3)], outline=INK, width=1)
    elif t == "sapling":
        dr.line([P(x - .6, y), P(x + .6, y)], fill=INK, width=1)
        dr.line([P(x, y - .6), P(x, y + .6)], fill=INK, width=1)

# nehir + set şevi + dalgalar; sulama kanalları + savaklar
dr.line([P(*p) for p in RIVER], fill=WATER, width=int(RIVER_W * S), joint="curve")
for sgn in (-1, 1):
    bank = []
    for x, y, a, s_ in resample(RIVER, 1.0):
        nx, ny = -math.sin(a) * sgn, math.cos(a) * sgn
        bank.append((x + nx * RIVER_W / 2, y + ny * RIVER_W / 2))
        if int(s_) % 2 == 0 and not at(road, x + nx * (RIVER_W / 2 + 1.2), y + ny * (RIVER_W / 2 + 1.2)):
            dr.line([P(x + nx * (RIVER_W / 2 + .3), y + ny * (RIVER_W / 2 + .3)),
                     P(x + nx * (RIVER_W / 2 + 2.2), y + ny * (RIVER_W / 2 + 2.2))], fill=GREY, width=1)
    dr.line([P(*p) for p in bank], fill=INK, width=3)
for x, y, a, s_ in resample(RIVER, 7.0):
    xx_ = x - math.sin(a) * (2.2 if int(s_ / 7) % 2 else -2.2)
    dr.arc([P(xx_ - 1.6, y - .8), P(xx_, y + .8)], 180, 360, fill=WATER_D, width=2)
    dr.arc([P(xx_, y - .8), P(xx_ + 1.6, y + .8)], 0, 180, fill=WATER_D, width=2)
for n, w, pts in CHANNELS:
    dr.line([P(*p) for p in pts], fill=WATER, width=max(3, int(w * S)), joint="curve")
    for sgn in (-1, 1):
        dr.line([P(x - math.sin(a) * sgn * w / 2, y + math.cos(a) * sgn * w / 2) for x, y, a, _ in resample(pts, 0.8)], fill=INK, width=1)
    for x, y, a, s_ in resample(pts, 9.0)[1:]:           # akış yönü
        dr.line([P(x - math.cos(a) * .8 - math.sin(a) * .5, y - math.sin(a) * .8 + math.cos(a) * .5), P(x, y),
                 P(x - math.cos(a) * .8 + math.sin(a) * .5, y - math.sin(a) * .8 - math.cos(a) * .5)], fill=WATER_D, width=1)
for x, y in WEIRS:
    dr.rectangle([P(x - 1.6, y - .5), P(x + 1.6, y + .5)], fill=INK)

# açık drenaj kanalı (eski mahalle ve tarla yolları boyunca)
for name, kind, wd, pts in ROADS:
    if name not in DITCH_ROADS:
        continue
    for sgn in (-1, 1):
        for off in (wd / 2 + 0.45, wd / 2 + 0.95):
            seg = []
            for x, y, a, _ in resample(pts, 0.6):
                ex, ey = x - math.sin(a) * sgn * off, y + math.cos(a) * sgn * off
                if at(road, ex, ey) or at(solid, ex, ey) or at(water, ex, ey) or not at(settle | fieldz, ex, ey):
                    if len(seg) > 1:
                        dr.line(seg, fill=GREY, width=1)
                    seg = []
                else:
                    seg.append(P(ex, ey))
            if len(seg) > 1:
                dr.line(seg, fill=GREY, width=1)

# yollar
paved = [r for r in ROADS if r[1] in ("main", "road", "lane", "alley")]
for n, k, w, pts in paved:
    dr.line([P(*p) for p in pts], fill=INK, width=int(w * S) + 5, joint="curve")
for n, k, w, pts in paved:
    dr.line([P(*p) for p in pts], fill=PAPER, width=int(w * S) - 1, joint="curve")
for n, k, w, pts in ROADS:
    if k == "dirt":
        for sgn in (-1, 1):
            for x, y, a, _ in resample(pts, 1.3):
                ex, ey = x - math.sin(a) * sgn * w / 2, y + math.cos(a) * sgn * w / 2
                if not (at(solid, ex, ey) or at(water, ex, ey) or at(taken, ex, ey)):
                    dr.ellipse([P(ex - .18, ey - .18), P(ex + .18, ey + .18)], fill=INK)
    elif k == "trail":                                   # bisiklet patikası: kesik-nokta çift kenar
        for i, (x, y, a, _) in enumerate(resample(pts, 1.0)):
            for sgn in (-1, 1):
                ex, ey = x - math.sin(a) * sgn * w / 2, y + math.cos(a) * sgn * w / 2
                if i % 4 in (0, 1) and not at(water, ex, ey):
                    dr.line([P(ex, ey), P(ex + math.cos(a) * .9, ey + math.sin(a) * .9)], fill=INK, width=2)
                elif i % 4 == 3 and not at(water, ex, ey):
                    dr.point(P(ex, ey), fill=INK)
    elif k == "post":                                    # eski posta patikası: tek kesik çizgi
        for i, (x, y, a, _) in enumerate(resample(pts, 1.2)):
            if i % 3 != 2:
                dr.line([P(x, y), P(x + math.cos(a) * .9, y + math.sin(a) * .9)], fill=INK, width=2)
# köprüler: Kaido köprüsü, kanal geçişleri (menfez köprü)
b0, b1 = M((318, 972)), M((440, 968))
dr.polygon([P(b0[0], b0[1] - 3.3), P(b1[0], b1[1] - 3.3), P(b1[0], b1[1] + 3.3), P(b0[0], b0[1] + 3.3)], fill=PAPER)
for off in (-3.6, 3.6):
    dr.line([P(b0[0], b0[1] + off), P(b1[0], b1[1] + off)], fill=INK, width=6)
for sx_ in (b0[0], b1[0]):
    for off in (-1, 1):
        dr.line([P(sx_, b0[1] + off * 3.6), P(sx_ + (1.5 if sx_ == b0[0] else -1.5) * -1, b0[1] + off * 5.2)], fill=INK, width=3)
chan_mask = grid()
for n, w, pts in CHANNELS:
    stamp_line(chan_mask, pts, w)
for name, kind, wd, pts in ROADS:
    if kind == "bridge":
        continue
    prev = False
    for x, y, a, _ in resample(pts, 0.5):
        on = at(chan_mask, x, y)
        if on and not prev:
            for sgn in (-1, 1):
                ox, oy = -math.sin(a) * sgn * (wd / 2 + 0.6), math.cos(a) * sgn * (wd / 2 + 0.6)
                dr.line([P(x + ox - math.cos(a) * 1.2, y + oy - math.sin(a) * 1.2), P(x + ox + math.cos(a) * 1.8, y + oy + math.sin(a) * 1.8)], fill=INK, width=3)
        prev = on
# demiryolu, demiryolu köprüsü, hemzemin geçitler (fumikiri)
for off in (-1.0, 1.0):
    dr.line([P(0, RAIL_Y + off), P(W, RAIL_Y + off)], fill=INK, width=3)
crossings = []
for name, kind, wd, pts in ROADS:
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 - RAIL_Y) * (y1 - RAIL_Y) < 0:
            t = (RAIL_Y - y0) / (y1 - y0)
            crossings.append((x0 + (x1 - x0) * t, wd, name))
xr = 0.0
while xr < W:
    if not any(abs(xr - cx) < w / 2 + 0.6 for cx, w, _ in crossings):
        dr.line([P(xr, RAIL_Y - 1.9), P(xr, RAIL_Y + 1.9)], fill=INK, width=2)
    xr += 2.6
for cx, w, _ in crossings:
    dr.rectangle([P(cx - w / 2, RAIL_Y - 2.2), P(cx + w / 2, RAIL_Y + 2.2)], fill=PAPER)
    for off in (-1.0, 1.0):
        dr.line([P(cx - w / 2, RAIL_Y + off), P(cx + w / 2, RAIL_Y + off)], fill=INK, width=3)
    for sgn in (-1, 1):                                  # geçit bariyerleri
        dr.line([P(cx - w / 2 - .6, RAIL_Y + sgn * 3.6), P(cx + w / 2 * 0.2, RAIL_Y + sgn * 3.6)], fill=INK, width=3)
        dr.ellipse([P(cx - w / 2 - 1.1, RAIL_Y + sgn * 3.6 - .5), P(cx - w / 2 - .1, RAIL_Y + sgn * 3.6 + .5)], fill=INK)
rb = M((368, 1392))[0]
dr.rectangle([P(rb, RAIL_Y - 2.6), P(rb + 17, RAIL_Y + 2.6)], fill=PAPER, outline=INK, width=3)
for k in range(1, 5):
    dr.line([P(rb + k * 3.4, RAIL_Y - 2.6), P(rb + k * 3.4, RAIL_Y + 2.6)], fill=INK, width=2)


# parseller: bahçe içerikleri, duvarlar, çitler, kapılar
def wall(p, q, style):
    if style == "stone":
        dr.line([P(*p), P(*q)], fill=INK, width=5)
    elif style == "block":
        dr.line([P(*p), P(*q)], fill=INK, width=3)
    elif style == "fence":
        dr.line([P(*p), P(*q)], fill=INK, width=1)
        L_ = math.hypot(q[0] - p[0], q[1] - p[1])
        if L_ > 0:
            ux, uy = (q[0] - p[0]) / L_, (q[1] - p[1]) / L_
            for d_ in np.arange(0.5, L_, 1.0):
                x, y = p[0] + ux * d_, p[1] + uy * d_
                dr.line([P(x - uy * .35, y + ux * .35), P(x + uy * .35, y - ux * .35)], fill=INK, width=1)
    elif style == "hedge":
        L_ = math.hypot(q[0] - p[0], q[1] - p[1])
        for d_ in np.arange(0.4, L_, 0.85):
            x, y = p[0] + (q[0] - p[0]) * d_ / max(L_, 1e-6), p[1] + (q[1] - p[1]) * d_ / max(L_, 1e-6)
            dr.ellipse([P(x - .42, y - .42), P(x + .42, y + .42)], outline=INK, width=1)
    else:
        dr.line([P(*p), P(*q)], fill=LIGHT, width=1)


for kind_, data in extras:
    if kind_ == "veg_rows":
        a_, b_, c_, d_ = data
        L_ = math.hypot(c_[0] - b_[0], c_[1] - b_[1])
        for f_ in np.arange(0.15, 1.0, max(0.12, 1.0 / max(L_, 1))):
            p0 = (a_[0] + (d_[0] - a_[0]) * f_, a_[1] + (d_[1] - a_[1]) * f_)
            p1 = (b_[0] + (c_[0] - b_[0]) * f_, b_[1] + (c_[1] - b_[1]) * f_)
            dr.line([P(*p0), P(*p1)], fill=LIGHT, width=1)
    elif kind_ == "courtyard":
        for i in range(0, 4):
            pass
        dr.polygon([P(*p) for p in data], outline=LIGHT, width=1)
    elif kind_ == "parking_pocket":
        dr.polygon([P(*p) for p in data], outline=INK, width=1)
        dr.line([P(*data[0]), P(*data[2])], fill=GREY, width=1)
for kind_, data in extras:
    if kind_ == "shrub":
        x, y = data
        dr.ellipse([P(x - .55, y - .55), P(x + .55, y + .55)], outline=INK, width=1)
    elif kind_ == "windbreak":
        broadleaf(*data, r=0.95)
    elif kind_ == "path":
        for i, (x, y, a, _) in enumerate(resample(data, 0.9)):
            dr.rectangle([P(x - .3, y - .22), P(x + .3, y + .22)], outline=GREY)
for pa in passages:
    for i, (x, y, a, _) in enumerate(resample(pa, 0.8)):
        if i % 2 == 0:
            dr.point(P(x, y), fill=INK)
for p in parcels:
    poly = p["poly"]
    st = p["wall"]
    if p["zone"] in ("mountain", "farm"):
        for i in range(4):
            wall(poly[i], poly[(i + 1) % 4], st)
        continue
    # yan ve arka sınırlar
    wall(poly[1], poly[2], st if st != "none" else "none")
    wall(poly[2], poly[3], st if st != "none" else "none")
    wall(poly[3], poly[0], st if st != "none" else "none")
    for seg in p["front"]:
        wall(seg[0], seg[1], st if p["zone"] != "shop" else "none")
    if p["gate"] and st in ("stone", "block", "fence", "hedge"):    # kapı babaları
        for g in p["gate"]:
            dr.rectangle([P(g[0] - .3, g[1] - .3), P(g[0] + .3, g[1] + .3)], fill=INK)

# binalar
SYMBOLS = ("torii", "tunnel", "water_tower", "radio_mast", "lookout")
for b in buildings:
    x, y, a = b["x"], b["y"], b["ang"]
    c = rect_corners(x, y, b["w"], b["d"], a)
    t = b["type"]
    if t == "torii":
        dr.line([P(*c[0]), P(*c[1])], fill=INK, width=5)
        dr.line([P(*c[3]), P(*c[2])], fill=INK, width=3)
        for f_ in (0.18, 0.82):
            px_ = c[0][0] + (c[1][0] - c[0][0]) * f_
            dr.ellipse([P(px_ - .45, y - .45), P(px_ + .45, y + .45)], fill=INK)
        continue
    if t == "tunnel":
        dr.pieslice([P(x - 3, y - 3), P(x + 3, y + 3)], 180, 360, fill=INK)
        dr.arc([P(x - 4.2, y - 4.2), P(x + 4.2, y + 4.2)], 180, 360, fill=INK, width=4)
        dr.line([P(x - 4.2, y), P(x + 4.2, y)], fill=INK, width=3)
        continue
    if t == "water_tower":
        dr.ellipse([P(x - 2.8, y - 2.8), P(x + 2.8, y + 2.8)], fill=PAPER, outline=INK, width=3)
        dr.ellipse([P(x - 1.2, y - 1.2), P(x + 1.2, y + 1.2)], outline=INK, width=1)
        continue
    if t == "radio_mast":
        dr.polygon([P(x, y - 6), P(x - 1.8, y + 2), P(x + 1.8, y + 2)], outline=INK, width=2)
        dr.line([P(x - 1.2, y - 1), P(x + 1.2, y - 1)], fill=INK, width=1)
        continue
    if t == "lookout":
        dr.chord([P(x - 3.4, y - 2.5), P(x + 3.4, y + 2.5)], 180, 360, outline=INK, width=2)
        continue
    dashed = t in ("carport", "open_shed")
    if dashed:
        for i in range(4):
            p0, p1 = c[i], c[(i + 1) % 4]
            for f_ in np.arange(0, 1, 0.2):
                dr.line([P(p0[0] + (p1[0] - p0[0]) * f_, p0[1] + (p1[1] - p0[1]) * f_),
                         P(p0[0] + (p1[0] - p0[0]) * (f_ + 0.11), p0[1] + (p1[1] - p0[1]) * (f_ + 0.11))], fill=INK, width=2)
        dr.line([P(*c[0]), P(*c[2])], fill=LIGHT, width=1)
        continue
    wpx = 3 if (b["label"] or t in ("kura",)) else 2
    if t == "porch":
        dr.polygon(wob(c, .02), fill=PAPER, outline=INK, width=1)
        continue
    dr.polygon(wob(c, .05), fill=PAPER, outline=INK, width=wpx)
    ra = a if b["w"] >= b["d"] else a + math.pi / 2
    L_ = max(b["w"], b["d"]) / 2 - 0.9
    if t in ("shed",):
        dr.line([P(*c[0]), P(*c[2])], fill=GREY, width=1)
    elif t == "kura":                                    # kura: kalın duvar, çift çatı
        ci_ = rect_corners(x, y, b["w"] - 1.0, b["d"] - 1.0, a)
        dr.polygon([P(*p) for p in ci_], outline=INK, width=1)
    elif t == "barn":
        dr.line([P(*c[0]), P(*c[2])], fill=INK, width=1)
        dr.line([P(*c[1]), P(*c[3])], fill=INK, width=1)
    elif t in ("apartment", "school", "school_wing", "market"):
        k_ = 3 if t == "apartment" else 6
        for i in range(1, k_):
            f_ = i / k_
            p0 = (c[0][0] + (c[1][0] - c[0][0]) * f_, c[0][1] + (c[1][1] - c[0][1]) * f_)
            p1 = (c[3][0] + (c[2][0] - c[3][0]) * f_, c[3][1] + (c[2][1] - c[3][1]) * f_)
            dr.line([P(*p0), P(*p1)], fill=GREY, width=1)
    else:
        dr.line([P(x - math.cos(ra) * L_, y - math.sin(ra) * L_), P(x + math.cos(ra) * L_, y + math.sin(ra) * L_)], fill=INK, width=2)
    if t == "akiya":
        dr.line([P(*c[0]), P(*c[2])], fill=GREY, width=1)
        dr.line([P(*c[1]), P(*c[3])], fill=GREY, width=1)
    if t == "shop":                                      # dükkân cephesi: saçak çizgisi
        dr.line([P(*c[0]), P(*c[1])], fill=INK, width=4) if True else None
    if t == "honden":                                    # tamagaki çit
        cf = rect_corners(x, y, b["w"] + 4, b["d"] + 4, a)
        for i in range(4):
            wall(cf[i], cf[(i + 1) % 4], "fence")
    if b["enterable"]:                                   # girilebilir iç mekân: çift çerçeve
        ci_ = rect_corners(x, y, b["w"] - 1.0, b["d"] - 1.0, a)
        dr.polygon([P(*p) for p in ci_], outline=INK, width=2)
# genkan, posta kutusu, teslimat numarası
for b in buildings:
    if "genkan" in b:
        gx, gy = b["genkan"]
        dr.polygon([P(gx - .55, gy - .55), P(gx + .55, gy - .55), P(gx + .55, gy + .55), P(gx - .55, gy + .55)], fill=INK)
    if "mailbox" in b:
        mx, my = b["mailbox"]
        dr.rectangle([P(mx - .3, my - .3), P(mx + .3, my + .3)], fill=PAPER, outline=INK, width=2)
for b in buildings:
    if b["delivery"]:
        if "mailbox" in b:
            mx, my = b["mailbox"]
        else:
            c = rect_corners(b["x"], b["y"], b["w"], b["d"], b["ang"])
            mx, my = c[0]
        r = 1.5
        dr.ellipse([P(mx - r, my - r - 1.6), P(mx + r, my + r - 1.6)], fill=INK)
        tt = str(b["delivery"])
        tw = dr.textlength(tt, font=F_NUM)
        dr.text((mx * S - tw / 2, (my - 1.6) * S - 6.5), tt, font=F_NUM, fill=PAPER)

# donatılar
for x, y in poles:
    dr.ellipse([P(x - .35, y - .35), P(x + .35, y + .35)], fill=INK)
for x, y in JIZO:
    dr.ellipse([P(x - .6, y - 1.2), P(x + .6, y)], outline=INK, width=2)
    dr.line([P(x - .8, y + .2), P(x + .8, y + .2)], fill=INK, width=2)
for x, y in LANTERNS:
    dr.rectangle([P(x - .5, y - .5), P(x + .5, y + .5)], outline=INK, width=2)
    dr.point(P(x, y), fill=INK)
dr.rectangle([P(BUS_STOP[0] - .7, BUS_STOP[1] - .7), P(BUS_STOP[0] + .7, BUS_STOP[1] + .7)], fill=INK)
dr.line([P(BUS_STOP[0], BUS_STOP[1] + .7), P(BUS_STOP[0], BUS_STOP[1] + 2.2)], fill=INK, width=2)

# yazılar
_boxes = []


def text(x, y, t, f=F_IT, color=INK):
    tw = dr.textlength(t, font=f)
    for dy in (0, 4, -4, 8, -8, 12):
        bx = (x * S - tw / 2, (y + dy) * S - f.size / 2, x * S + tw / 2, (y + dy) * S + f.size / 2)
        if not any(bx[0] < o[2] and o[0] < bx[2] and bx[1] < o[3] and o[1] < bx[3] for o in _boxes):
            break
    _boxes.append(bx)
    dr.text(bx[:2], t, font=f, fill=color, stroke_width=4, stroke_fill=PAPER)


def rot_text(x, y, t, a, f=F_SM):
    tw = dr.textlength(t, font=f) + 8
    tmp = Image.new("RGBA", (int(tw), f.size + 8), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    td.text((4, 3), t, font=f, fill=INK, stroke_width=3, stroke_fill=PAPER)
    deg = -math.degrees(a)
    deg = deg - 180 if deg > 90 else deg + 180 if deg < -90 else deg
    tmp = tmp.rotate(deg, expand=True, resample=Image.BICUBIC)
    img.paste(tmp, (int(x * S - tmp.width / 2), int(y * S - tmp.height / 2)), tmp)


# yol genişliği ölçüleri
for name, kind, wd, pts in ROADS:
    if name not in WIDTH_NOTES:
        continue
    smp = resample(pts, 0.5)
    x, y, a, _ = smp[int(len(smp) * WIDTH_NOTES[name])]
    nx, ny = -math.sin(a), math.cos(a)
    p0, p1 = (x - nx * wd / 2, y - ny * wd / 2), (x + nx * wd / 2, y + ny * wd / 2)
    dr.line([P(*p0), P(*p1)], fill=INK, width=1)
    for q in (p0, p1):
        dr.line([P(q[0] - math.cos(a) * .7, q[1] - math.sin(a) * .7), P(q[0] + math.cos(a) * .7, q[1] + math.sin(a) * .7)], fill=INK, width=1)
    lx, ly = x + nx * (wd / 2 + 2.2), y + ny * (wd / 2 + 2.2)
    rot_text(lx, ly, f"{wd:.1f} m", a, F_XS)
for name in ("Old Kaido", "Chuo-dori", "East Road", "River Road", "Mountain Road", "Shotengai", "Station Front", "Sakamichi",
             "Forestry road", "Old postal path", "Bike trail"):
    for n, k, w, pts in ROADS:
        if n == name:
            smp = resample(pts, 1.0)
            x, y, a, _ = smp[int(len(smp) * (0.62 if name != "Shotengai" else 0.7))]
            rot_text(x, y, name.upper() if k in ("main", "road", "lane") else name, a)
for b in buildings:
    if b["label"]:
        lab = b["label"]
        dy = b["d"] / 2 + 2.4
        text(b["x"], b["y"] - dy, lab, F_SM if b["type"] in RESID | {"shop"} else F_IT)
for a in areas:
    if a["label"]:
        x0, y0, x1, y1 = a["rect"]
        text((x0 + x1) / 2, (y0 + y1) / 2, a["label"], F_XS)
for ref, t, f in [((1150, 175), "S U G I   F O R E S T", F_BIG), ((690, 705), "F U R U M A C H I", F_BIG), ((690, 735), "(old quarter)", F_IT),
                  ((1560, 1118), "east road district", F_IT), ((1760, 1290), "FARM BELT", F_BIG),
                  ((800, 1468), "RICE PADDIES", F_BIG), ((268, 700), "KUZU RIVER", F_BIG), ((128, 1430), "canal", F_IT),
                  ((150, 1000), "To Kamiyama (6 km)", F_IT), ((240, 1375), "<- to Terminal City", F_IT),
                  ((1780, 1378), "to Nagiri Jct. (22 km) ->", F_IT), ((1180, 735), "vegetable plots", F_IT),
                  ((1820, 150), "Old tunnel (sealed)", F_IT), ((905, 480), "stone steps +16 m", F_SM), ((1000, 300), "sacred grove", F_SM),
                  ((1300, 195), "clear-cut", F_XS), ((1695, 640), "clear-cut", F_XS), ((605, 420), "clear-cut", F_XS),
                  ((1250, 250), "sugi plantation", F_SM), ((1730, 700), "sugi plantation", F_SM), ((440, 1010), "jizo", F_XS),
                  ((1150, 1310), "bus", F_XS), ((410, 600), "weir", F_XS), ((1004, 1430), "fumikiri", F_XS)]:
    text(*M(ref), t, f)

# başlık kartuşu
tx0, ty0 = M((110, 110))
tx1, ty1 = M((730, 280))
dr.rectangle([P(tx0, ty0), P(tx1, ty1)], fill=PAPER, outline=INK, width=4)
dr.rectangle([P(tx0 + 1.2, ty0 + 1.2), P(tx1 - 1.2, ty1 - 1.2)], outline=INK, width=2)
dr.text(P(tx0 + 7, ty0 + 4), "MIZUNAGI-MACHI", font=F_TITLE, fill=INK)
dr.text((P(tx0 + 7, ty0)[0] + dr.textlength("MIZUNAGI-MACHI", font=F_TITLE) + 14, P(0, ty0 + 6)[1]), "水凪町", font=F_JP, fill=INK)
dr.text(P(tx0 + 8, ty0 + 21), "Postal delivery area · 1991 · Kiso valley, Nagano", font=F_IT, fill=INK)
dr.text(P(tx0 + 8, ty0 + 28), f"Settlement plan, top view · {W:.0f} x {H:.0f} m", font=F_SM, fill=INK)
dr.rectangle([0, 0, img.width - 1, MAP_H - 1], outline=INK, width=8)

# ------------------------------------------------------------ açıklama kutusu (ayrı bant)
LY = MAP_H + 18
dr.rectangle([12, LY, img.width - 12, img.height - 14], outline=INK, width=4)
dr.text((32, LY + 12), "LEGEND", font=F_BIG, fill=INK)
cols = [32, 380, 728, 1076]
ROW = 30


def lg(col, row, draw_fn, label):
    x0, y0 = cols[col], LY + 52 + row * ROW
    draw_fn(x0, y0 + 10)
    dr.text((x0 + 70, y0), label, font=F_SM, fill=INK)


def L_road(w, k):
    def f(x, y):
        if k == "paved":
            dr.line([(x, y), (x + 56, y)], fill=INK, width=w + 5)
            dr.line([(x, y), (x + 56, y)], fill=PAPER, width=w - 1)
        elif k == "dirt":
            for xx_ in range(x, x + 56, 5):
                dr.ellipse([xx_ - 1, y - w // 2 - 1, xx_ + 1, y - w // 2 + 1], fill=INK)
                dr.ellipse([xx_ - 1, y + w // 2 - 1, xx_ + 1, y + w // 2 + 1], fill=INK)
        elif k == "trail":
            for xx_ in range(x, x + 56, 16):
                for sgn in (-1, 1):
                    dr.line([(xx_, y + sgn * 6), (xx_ + 8, y + sgn * 6)], fill=INK, width=2)
                    dr.point((xx_ + 12, y + sgn * 6), fill=INK)
        elif k == "post":
            for xx_ in range(x, x + 56, 7):
                dr.line([(xx_, y), (xx_ + 4, y)], fill=INK, width=2)
    return f


def L_wall(st):
    def f(x, y):
        wall((x / S, y / S), ((x + 56) / S, y / S), st)
    return f


def L_box(fill=PAPER, dash=False, ridge=True, double=False, cross=False):
    def f(x, y):
        dr.rectangle([x + 8, y - 9, x + 48, y + 9], fill=fill, outline=INK, width=2)
        if ridge:
            dr.line([(x + 12, y), (x + 44, y)], fill=INK, width=2)
        if double:
            dr.rectangle([x + 12, y - 5, x + 44, y + 5], outline=INK, width=2)
        if cross:
            dr.line([(x + 8, y - 9), (x + 48, y + 9)], fill=GREY, width=1)
            dr.line([(x + 48, y - 9), (x + 8, y + 9)], fill=GREY, width=1)
    return f


def L_dash(x, y):
    for i in range(0, 40, 8):
        dr.line([(x + 8 + i, y - 9), (x + 12 + i, y - 9)], fill=INK, width=2)
        dr.line([(x + 8 + i, y + 9), (x + 12 + i, y + 9)], fill=INK, width=2)
    dr.line([(x + 8, y - 9), (x + 8, y + 9)], fill=INK, width=2)
    dr.line([(x + 48, y - 9), (x + 48, y + 9)], fill=INK, width=2)


def L_water(w):
    def f(x, y):
        dr.line([(x, y), (x + 56, y)], fill=WATER, width=w)
        dr.line([(x, y - w // 2), (x + 56, y - w // 2)], fill=INK, width=1)
        dr.line([(x, y + w // 2), (x + 56, y + w // 2)], fill=INK, width=1)
    return f


def L_rail(x, y):
    dr.line([(x, y - 4), (x + 56, y - 4)], fill=INK, width=3)
    dr.line([(x, y + 4), (x + 56, y + 4)], fill=INK, width=3)
    for xx_ in range(x, x + 56, 10):
        dr.line([(xx_, y - 8), (xx_, y + 8)], fill=INK, width=2)


def L_cross(x, y):
    L_rail(x, y)
    dr.rectangle([x + 18, y - 9, x + 38, y + 9], fill=PAPER)
    dr.line([(x + 18, y - 4), (x + 38, y - 4)], fill=INK, width=3)
    dr.line([(x + 18, y + 4), (x + 38, y + 4)], fill=INK, width=3)
    dr.line([(x + 14, y - 13), (x + 28, y - 13)], fill=INK, width=3)
    dr.line([(x + 14, y + 13), (x + 28, y + 13)], fill=INK, width=3)


def L_bridge(x, y):
    dr.line([(x, y), (x + 56, y)], fill=WATER, width=16)
    dr.rectangle([x + 16, y - 12, x + 40, y + 12], fill=PAPER)
    dr.line([(x + 16, y - 12), (x + 16, y + 12)], fill=INK, width=4)
    dr.line([(x + 40, y - 12), (x + 40, y + 12)], fill=INK, width=4)


def L_steps(x, y):
    dr.rectangle([x + 4, y - 7, x + 52, y + 7], outline=INK, width=2)
    for xx_ in range(x + 8, x + 52, 4):
        dr.line([(xx_, y - 7), (xx_, y + 7)], fill=INK, width=1)


def L_cont(x, y):
    dr.arc([x, y - 6, x + 56, y + 18], 200, 340, fill=(200, 193, 177), width=2)
    dr.arc([x + 6, y - 2, x + 50, y + 22], 200, 340, fill=(200, 193, 177), width=2)


def L_trees(kind):
    def f(x, y):
        for i in range(4):
            xx_ = (x + 8 + i * 13) / S
            if kind == "plant":
                tri(xx_, (y + 9) / S, 3.2, 1.15)
            elif kind == "nat":
                if i % 2:
                    broadleaf(xx_, (y + 3) / S, 1.0)
                else:
                    tri(xx_, (y + 9) / S, 5.0, 1.8)
            elif kind == "clear":
                dr.ellipse([xx_ * S - 2, y - 2, xx_ * S + 2, y + 2], outline=INK)
            elif kind == "grove":
                if i < 3:
                    broadleaf(xx_ + .6 * i, (y + 2) / S, 1.5)
    return f


def L_paddy(x, y):
    dr.rectangle([x + 4, y - 10, x + 52, y + 10], outline=GREY, width=2)
    for k in (-5, 0, 5):
        dr.line([(x + 8, y + k), (x + 48, y + k)], fill=(170, 192, 200), width=1)


def L_veg(x, y):
    dr.rectangle([x + 4, y - 10, x + 52, y + 10], outline=GREY, width=2)
    for xx_ in range(x + 8, x + 50, 4):
        dr.line([(xx_, y - 7), (xx_, y + 7)], fill=LIGHT, width=1)


def L_gate(x, y):
    wall((x / S, y / S), ((x + 18) / S, y / S), "block")
    wall(((x + 34) / S, y / S), ((x + 56) / S, y / S), "block")
    for gx in (x + 18, x + 34):
        dr.rectangle([gx - 2, y - 2, gx + 2, y + 2], fill=INK)
    dr.rectangle([x + 38, y + 4, x + 44, y + 10], fill=PAPER, outline=INK, width=2)


def L_genkan(x, y):
    dr.rectangle([x + 8, y - 9, x + 48, y + 9], fill=PAPER, outline=INK, width=2)
    dr.rectangle([x + 24, y + 5, x + 32, y + 13], fill=INK)


def L_num(x, y):
    dr.ellipse([x + 20, y - 9, x + 38, y + 9], fill=INK)
    dr.text((x + 24, y - 7), "12", font=F_NUM, fill=PAPER)


def L_pole(x, y):
    dr.ellipse([x + 26, y - 2, x + 30, y + 2], fill=INK)


def L_weir(x, y):
    L_water(8)(x, y)
    dr.rectangle([x + 22, y - 6, x + 34, y + 6], fill=INK)


def L_ditch(x, y):
    dr.line([(x, y - 3), (x + 56, y - 3)], fill=GREY, width=1)
    dr.line([(x, y + 3), (x + 56, y + 3)], fill=GREY, width=1)


def L_torii(x, y):
    dr.line([(x + 14, y - 3), (x + 42, y - 3)], fill=INK, width=5)
    dr.line([(x + 16, y + 3), (x + 40, y + 3)], fill=INK, width=3)


def L_parking(x, y):
    dr.rectangle([x + 4, y - 10, x + 52, y + 10], outline=INK, width=2)
    for xx_ in range(x + 14, x + 50, 10):
        dr.line([(xx_, y - 10), (xx_, y - 2)], fill=GREY, width=1)


LEG = [
    [(L_road(28, "paved"), "Road 6-7 m (asphalt)"), (L_road(18, "paved"), "Lane 4-5 m"), (L_road(11, "paved"), "Roji / alley 2.7 m"),
     (L_road(14, "dirt"), "Farm / forestry road 3-3.6 m"), (L_road(12, "trail"), "Bike trail / levee path 3 m"),
     (L_road(0, "post"), "Old postal path"), (L_rail, "Railway"), (L_cross, "Level crossing (fumikiri)"), (L_bridge, "Bridge / culvert"),
     (L_steps, "Stone steps")],
    [(L_water(14), "River"), (L_water(7), "Irrigation channel"), (L_weir, "Weir (intake)"), (L_ditch, "Open drainage ditch"),
     (L_paddy, "Rice paddy"), (L_veg, "Vegetable plot / home garden"), (L_cont, "Contour, 2 m"), (L_trees("nat"), "Natural mixed forest"),
     (L_trees("plant"), "Sugi plantation (rows)"), (L_trees("clear"), "Clear-cut")],
    [(L_wall("stone"), "Stone wall"), (L_wall("block"), "Concrete block wall"), (L_wall("fence"), "Wooden fence"),
     (L_wall("hedge"), "Hedge"), (L_wall("none"), "Parcel boundary"), (L_gate, "Gate + mailbox"), (L_genkan, "Building + genkan (entrance)"),
     (L_box(ridge=False, cross=True), "Storehouse / shed / barn"), (L_dash, "Carport / open shed"), (L_parking, "Parking")],
    [(L_box(cross=True), "Empty house (akiya)"), (L_box(double=True), "Enterable interior"), (L_num, "Active delivery address"),
     (L_torii, "Torii"), (L_trees("grove"), "Sacred grove"), (L_pole, "Utility pole"), (None, ""), (None, ""), (None, ""), (None, "")],
]
for ci_, col in enumerate(LEG):
    for ri, (fn, lab) in enumerate(col):
        if fn:
            lg(ci_, ri, fn, lab)
# kuzey oku + ölçek + özet (açıklama kutusunun sağ alt köşesi)
nx_, ny_ = 1340, LY + 260
dr.ellipse([nx_ - 34, ny_ - 34, nx_ + 34, ny_ + 34], outline=INK, width=3)
dr.polygon([(nx_, ny_ - 30), (nx_ - 11, ny_ + 14), (nx_, ny_ + 6), (nx_ + 11, ny_ + 14)], fill=INK)
dr.text((nx_ - 8, ny_ - 62), "N", font=F_BIG, fill=INK)
sx, sy = 1440, LY + 250
for i in range(5):
    dr.rectangle([sx + i * 10 * S / 1, sy, sx + (i + 1) * 10 * S, sy + 10], fill=INK if i % 2 == 0 else PAPER, outline=INK, width=2)
for i in range(0, 6):
    t = f"{i * 10}"
    dr.text((sx + i * 10 * S - dr.textlength(t, font=F_XS) / 2, sy + 14), t, font=F_XS, fill=INK)
dr.text((sx + 5 * 10 * S + 8, sy - 2), "m", font=F_SM, fill=INK)
sm = (f"{report['houses']} houses + {report['shophouses']} shop-houses · {report['delivery_addresses']} delivery addresses · "
      f"{report['enterable']} enterable interiors")
dr.text((1076, LY + 330), sm, font=F_SM, fill=INK)
dr.text((1076, LY + 352), "Districts read by parcel grain: Furumachi > Old Kaido > Shotengai > Core > East Road > Farm belt > Mountain",
        font=F_XS, fill=INK)

img.save(os.path.join(OUT, "mizunagi_plan.png"))


def r2(p):
    return [round(p[0], 2), round(p[1], 2)]


json.dump(dict(name="Mizunagi-machi", year=1991, scale_m_per_ref_px=K, size_m=[W, round(H, 2)], render_px_per_m=S, qa=report,
               buildings=buildings,
               parcels=[dict(id=p["id"], zone=p["zone"], road=p.get("road"), wall=p["wall"], built=p["built"], poly=[r2(q) for q in p["poly"]],
                             gate=[r2(q) for q in p["gate"]] if p["gate"] else None) for p in parcels],
               roads=[dict(name=n, kind=k, width_m=w, points=[r2(q) for q in pts]) for n, k, w, pts in ROADS],
               areas=areas, river=dict(width_m=RIVER_W, points=[r2(q) for q in RIVER]),
               channels=[dict(name=n, width_m=w, points=[r2(q) for q in pts]) for n, w, pts in CHANNELS], weirs=[r2(q) for q in WEIRS],
               rail_y=RAIL_Y, level_crossings=[dict(x=round(c[0], 2), road=c[2]) for c in crossings],
               fields=[dict(kind=k, poly=[r2(q) for q in f]) for k, f in FIELDS],
               plantations=[[r2(q) for q in p] for p, _ in PLANTATIONS], clearcuts=[[r2(q) for q in p] for p in CLEARCUTS],
               utility_poles=poles, jizo=[r2(q) for q in JIZO], bus_stop=r2(BUS_STOP), trees=len(trees)),
          open(os.path.join(OUT, "mizunagi_plan.json"), "w"), ensure_ascii=False, indent=1)
print("tamam ->", OUT)
