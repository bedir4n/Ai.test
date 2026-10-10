#!/usr/bin/env python3
"""Mizunagi-machi yükseklik haritası (Unreal Landscape için 16-bit).

Girdi : out/mizunagi_plan.json   (generate.py çıktısı: yollar, parseller, binalar, nehir, kanallar, tarlalar)
Çıktı : out/height/
          mizunagi_height.png      16-bit gri, 1765 x 1387  (Unreal önerilen boyut: 63x28 / 63x22 bileşen +1)
          mizunagi_height.r16      aynı veri, ham little-endian
          layer_*.png              8-bit katman ağırlıkları (asfalt, toprak yol, çakıl, çeltik, su yatağı, bahçe, orman, çim)
          preview_*.png            gölgelendirilmiş önizlemeler
          height_meta.json         Unreal import ayarları + QA raporu

Hücre 25 cm. Sıra (her aşama önceki aşamanın yüzeyini kilitler, üzerine yazmaz):
  1. doğal arazi (1 m): domain-warp fBm + sırt gürültüsü + akış erozyonu (stream-power)
  2. nehir: yukarıdan aşağı monoton yatak, taş şev, set (levee) kreti ve iç şev
  3. demiryolu: %0.8 eğim sınırı, balast setli yatak
  4. mabet: doğal sırta oturan düz teras, sahanlıklı taş basamak rampası
  5. yollar (öncelik sırası): boyuna profil = araziden yumuşatılmış + eğim sınırı + kavşak sabitleme,
     enine %2 bombe, 0.4 m banket, kesme 1:1 / dolgu 1:1.5 şev
  6. sulama kanalları: akış yönünde monoton, trapez kesit; yol altında menfez (kesilmez)
  7. çeltik: her tarla düz teras, 25 cm sırt (aze); bostan hafif düzleme
  8. alanlar ve parseller: düz ped, ön kenar yol kotunun 12 cm üstü, şevli/istinatlı geçiş
"""
import json, math, os, time
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = json.load(open(os.path.join(HERE, "out", "mizunagi_plan.json")))
OUT = os.path.join(HERE, "out", "height")
os.makedirs(OUT, exist_ok=True)

CELL = 0.25                                 # m / piksel  (Unreal X/Y ölçeği 25)
NX, NY = 63 * 28 + 1, 63 * 22 + 1           # 1765 x 1387 -> 441 x 346.5 m
rng = np.random.default_rng(1991)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a)


# ------------------------------------------------------------------ yardımcılar
def resample(pts, step):
    pts = np.asarray(pts, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(s[-1] / step) + 1)
    ss = np.linspace(0, s[-1], n)
    return np.stack([np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])], 1), ss


def sample(Z, xy, order=1):
    return ndi.map_coordinates(Z, [xy[:, 1] / CELL, xy[:, 0] / CELL], order=order, mode="nearest")


def bbox_of(xy, pad):
    x0 = max(0, int((xy[:, 0].min() - pad) / CELL))
    x1 = min(NX, int((xy[:, 0].max() + pad) / CELL) + 2)
    y0 = max(0, int((xy[:, 1].min() - pad) / CELL))
    y1 = min(NY, int((xy[:, 1].max() + pad) / CELL) + 2)
    return x0, x1, y0, y1


def poly_local(poly, bb):
    x0, x1, y0, y1 = bb
    im = Image.new("1", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(im).polygon([(x / CELL - x0, y / CELL - y0) for x, y in poly], fill=1)
    return np.array(im, bool)


def smooth1d(z, sigma_samples, pins=None):
    if sigma_samples <= 0:
        return z
    out = ndi.gaussian_filter1d(z, sigma_samples, mode="nearest")
    if pins is not None:
        out[pins] = z[pins]
    return out


def grade_limit(z, ds, g, pins, iters=60):
    """Boyuna eğimi |dz/ds| <= g olacak şekilde sınırla; sabit noktalar (kavşaklar) korunur."""
    z = z.copy()
    lim = g * ds
    for _ in range(iters):
        z0 = z.copy()
        for i in range(1, len(z)):
            z[i] = min(max(z[i], z[i - 1] - lim), z[i - 1] + lim)
        z[pins] = z0[pins] if _ == 0 else PIN_VALS[pins]
        for i in range(len(z) - 2, -1, -1):
            z[i] = min(max(z[i], z[i + 1] - lim), z[i + 1] + lim)
        z[pins] = PIN_VALS[pins]
        if np.abs(z - z0).max() < 1e-4:
            break
    return z


PIN_VALS = None


# ------------------------------------------------------------------ 1) doğal arazi (1 m ızgara)
def fbm(shape, scales, amps, seed):
    r = np.random.default_rng(seed)
    out = np.zeros(shape)
    for sc, am in zip(scales, amps):
        gy, gx = int(shape[0] / sc) + 4, int(shape[1] / sc) + 4
        lat = r.standard_normal((gy, gx))
        yy = np.arange(shape[0]) / sc + 1
        xx = np.arange(shape[1]) / sc + 1
        out += am * ndi.map_coordinates(lat, np.meshgrid(yy, xx, indexing="ij"), order=3, mode="reflect")
    return out


BW, BH = 442, 348
Y1, X1 = np.mgrid[0:BH, 0:BW].astype(float)
warp_x = fbm((BH, BW), (90, 40), (9, 4), 1)
warp_y = fbm((BH, BW), (90, 40), (9, 4), 2)
WX, WY = X1 + warp_x, Y1 + warp_y


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# --- makro şekil: Kiso referansına göre kalibre (yamaç gövdesi ~28-32°, etekte içbükey kolüvyal geçiş)
def slope_profile(d, H, L, fillet=6.0):
    """Etekten uzaklık d (m) -> yükseklik. Net dağ eteği: ~6 m'lik içbükey geçişten sonra hemen dik (H/L),
    sırta doğru yatıklaşır (Kiso vadisi yamaç önü)."""
    d = np.clip(d, 0, None)
    dd = np.where(d < fillet, d * d / (2 * fillet), d - fillet / 2)
    return H * (1 - np.exp(-dd / L))


floor = 0.5 + 0.010 * np.clip(318 - WY, 0, None) + 0.006 * np.clip(WX - 240, 0, None)      # vadi tabanı (taraça): ~%1 eğim
foot_n = 142 - 8 * np.sin(WX / 55) - 5 * np.sin(WX / 23 + 1.3) - 22 * np.exp(-((WX - 215) / 35) ** 2)   # kuzey etek; mabet burnu güneye uzanır
north = slope_profile(foot_n - WY, 80, 115)
foot_e = 345 + 14 * np.sin(WY / 41 + 0.7)
east = slope_profile(WX - foot_e, 65, 100) * sstep(262, 200, WY)
west = slope_profile(62 - WX, 40, 60) * sstep(185, 120, WY)
hi, lo = np.maximum(north, east), np.minimum(north, east)
mac = hi + 0.25 * lo + 0.6 * west                                                        # yumuşak maksimum: köşede yığılma yok
south_hill = 3 * sstep(330, 346, WY)
mtn = np.clip(mac / 22, 0, 1)

# --- gerçek Kiso referansı (yalnız eğim kalibrasyonu ve QA karşılaştırması için)
REF = np.load(os.path.join(HERE, "ref", "kiso_reference.npz"))
DEM, RES = REF["narai"].astype(np.float64), float(REF["res_m"])
REF_INFO = dict(source="AWS Terrain Tiles (terrarium z15), Narai-juku Kiso", res_m=round(RES, 2), window_north=[300, 300, 0, False])

# --- hidrolik erozyon (damla tabanlı, Gaea/World Machine yöntemi), iki ölçek:
#     2 m ızgara: ana dereler ~15-25 m aralıklı (gerçek birinci derece dere aralığı)
#     1 m ızgara: hafif ikinci geçiş, yamaç olukları
from erosion import droplets
log("hidrolik erozyon")
# nehir vadisi: taşkın düzlüğü biraz alçak
riv_xy, _ = resample(PLAN["river"]["points"], 1.0)
rv = np.zeros((BH, BW), bool)
ix = np.clip(riv_xy[:, 0].astype(int), 0, BW - 1)
iy = np.clip(riv_xy[:, 1].astype(int), 0, BH - 1)
rv[iy, ix] = True
dr_ = ndi.distance_transform_edt(~rv)
# nehir vadisi: dağ kesiminde nehir seviyesi yumuşak iner, vadi yamaçları ~31° (gerçek Kiso vadi duvarı)
near_i = ndi.distance_transform_edt(~rv, return_indices=True)[1]
riv_lvl_pts = np.interp(riv_xy[:, 1], [-10, 120, 346], [7.0, 2.5, -1.0])
lvl_img = np.full((BH, BW), np.nan)
lvl_img[iy, ix] = riv_lvl_pts
lvl = lvl_img[near_i[0], near_i[1]]
pre = floor + mac + south_hill + fbm((BH, BW), (120, 60, 25), (5.0, 2.5, 1.0), 21) * mtn ** 1.2
pre -= 1.4 * sstep(40, 8, dr_) * (1 - mtn)
pre = np.minimum(pre, lvl + 0.6 * np.clip(dr_ - 11, 0, None) + 1.2)                 # nehir vadisi erozyondan önce (yamaçları da oyulur)
c2 = ndi.zoom(pre, 0.5, order=1)
s2 = ndi.zoom(mac, 0.5, order=1) > 1.0
e2 = droplets(c2 / 120.0, s2, np.zeros(c2.shape, np.bool_), int(c2.size * 1.2), 7, radius=3) * 120.0
d2 = ndi.zoom(e2 - c2, 2.0, order=3)
d2 = np.pad(d2, ((0, max(0, BH - d2.shape[0])), (0, max(0, BW - d2.shape[1]))), mode="edge")[:BH, :BW]
e1 = pre + d2
e1 = droplets(e1 / 60.0, mac > 1.0, np.zeros(e1.shape, np.bool_), int(e1.size * 0.3), 11, radius=2) * 60.0
_dz = e1 - pre                                                            # etekte tortu yığılmasını sınırla: net dağ eteği
e1 = pre + np.where(_dz > 0, _dz * sstep(1.0, 14.0, mac), _dz)
from erosion import _clip_slope
for _ in range(6):                                                        # eşik yamaç: ~40° üstü kayar (talus)
    e1 = _clip_slope(e1, mac <= 0.3, math.tan(math.radians(40)))
e1 = np.where(mac > 0.3, ndi.gaussian_filter(e1, 0.6), e1)
detail = fbm((BH, BW), (60, 25, 10), (0.25, 0.12, 0.06), 4)
base = e1 + detail * (1 - mtn)
kd = next(a for a in PLAN["areas"] if a["type"] == "keidai")["rect"]
kcx, kcy = (kd[0] + kd[2]) / 2, (kd[1] + kd[3]) / 2
log("doğal arazi (gerçek Kiso dokusu)")
YY, XX = np.mgrid[0:NY, 0:NX].astype(np.float32) * CELL
Z = ndi.map_coordinates(base, [YY, XX], order=3, mode="nearest").astype(np.float64)
mtn_f = ndi.map_coordinates(mtn, [YY, XX], order=1, mode="nearest")
Z += fbm((NY, NX), (6, 3, 1.5), (0.04, 0.02, 0.01), 7) * (0.3 + 1.0 * mtn_f)
# --- mabet burnu: teras ve basamak hattı doğal bir sırt burnunun sırtında (yanlar ~35°, arkası yamaca bağlanır)
_sando_end = next(r for r in PLAN["roads"] if r["name"] == "Sando")["points"][-1]
_st = next(a for a in PLAN["areas"] if a["type"] == "steps")["rect"]
_foot = float(ndi.map_coordinates(Z, [[(_st[3] + 3.0) / CELL], [_sando_end[0] / CELL]], order=1)[0])
_kz = Z[int(kd[1] / CELL):int(kd[3] / CELL), int(kd[0] / CELL):int(kd[2] / CELL)]
_T = _foot + 16.0                                                     # tasarım: 16 m yükselti (~100 basamak); arkası taş duvarlı kazı
NATURAL = Z.copy()

lock = np.zeros((NY, NX), bool)              # kilitli (yapılmış) yüzey
SURF = []                                     # (bbox, maske, yüzey) — son aşamada öncelik sırasıyla yeniden basılır
LAYER = {k: np.zeros((NY, NX), np.float32) for k in ("asphalt", "dirt", "gravel", "paddy", "riverbed", "garden")}
water = np.zeros((NY, NX), bool)


# ------------------------------------------------------------------ koridor (yol, demiryolu) uygulayıcı
def _corridor_chunk(xy, prof, hw, crown, layer, skip, kc, kf, B, write_surface, shoulder, first=True, last=True):
    bb = bbox_of(xy, hw + shoulder + B)
    x0, x1, y0, y1 = bb
    lab = np.zeros((y1 - y0, x1 - x0), np.int32)
    lx = np.clip((xy[:, 0] / CELL).astype(int) - x0, 0, x1 - x0 - 1)
    ly = np.clip((xy[:, 1] / CELL).astype(int) - y0, 0, y1 - y0 - 1)
    lab[ly, lx] = np.arange(len(xy)) + 1
    dist, (iy_, ix_) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    sid = lab[iy_, ix_] - 1
    # pikseli eksene izdüşür: yay parametresi ve dik mesafe (kuantizasyon hatası yok)
    tg = np.gradient(xy, axis=0)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-12
    PY, PX = np.mgrid[y0:y1, x0:x1].astype(np.float64) * CELL
    rx_, ry_ = PX - xy[sid, 0], PY - xy[sid, 1]
    along = rx_ * tg[sid, 0] + ry_ * tg[sid, 1]
    d = np.abs(-rx_ * tg[sid, 1] + ry_ * tg[sid, 0])
    far = dist * CELL > hw + shoulder + 0.5
    d = np.where(far, dist * CELL, d)
    sidx = np.clip(sid + along / 0.25, 0, len(xy) - 1)
    Hc = np.interp(sidx, np.arange(len(xy)), prof)
    sub = Z[y0:y1, x0:x1]
    lk = lock[y0:y1, x0:x1]
    sk = skip[y0:y1, x0:x1] if skip is not None else np.zeros_like(lk)
    # parça uçlarında yuvarlak başlık oluşmasın: uç örneğe en yakın pikseller yalnızca gerçek yol ucunda yazılır
    endcap = ((sid == 0) & (not first)) | ((sid == len(xy) - 1) & (not last))
    sk = sk | endcap
    edge = Hc - crown * hw
    surf = d <= hw
    if write_surface:
        SURF.append((bb, (d <= hw + shoulder) & ~sk, np.where(surf, Hc - crown * d, edge)))
        m = surf & ~lk & ~sk
        sub[m] = (Hc - crown * d)[m]
        sh = (d > hw) & (d <= hw + shoulder) & ~lk & ~sk
        sub[sh] = edge[sh]
        lk |= m | sh
        if layer:
            LAYER[layer][y0:y1, x0:x1][m | sh] = 1.0
    t = d - hw - shoulder
    emb = (t > 0) & (t <= B) & ~lk & ~sk
    diff = sub - edge
    sub[emb] = (edge + np.clip(diff, -kf * t, kc * t))[emb]


def corridor(pts, hw, crown, grade, sigma_m, layer, skip=None, kc=1.0, kf=0.667, B=20.0, prof_override=None,
             write_surface=True, shoulder=0.4):
    global PIN_VALS
    xy, ss = resample(pts, 0.25)
    ds = ss[1] - ss[0] if len(ss) > 1 else 0.25
    zc = sample(Z, xy)
    ixs = np.clip((xy[:, 0] / CELL).astype(int), 0, NX - 1)
    iys = np.clip((xy[:, 1] / CELL).astype(int), 0, NY - 1)
    pins = lock[iys, ixs].copy()
    if skip is not None:
        pins &= ~skip[iys, ixs]
    if skip is not None:
        on = skip[iys, ixs]
        if on.any() and (~on).any():
            zc[on] = np.interp(ss[on], ss[~on], zc[~on])              # nehir üstü: köprü kotu doğrusal
    PIN_VALS = zc.copy()
    if prof_override is not None:
        prof = prof_override(ss, zc, pins)
    else:
        prof = smooth1d(zc, sigma_m / ds, pins)
        prof = grade_limit(prof, ds, grade, pins)
        prof = smooth1d(prof, 1.5 / ds, pins)
        prof = grade_limit(prof, ds, grade * 0.97, pins, iters=80)
    n = len(xy)
    CH = 60                                                   # 15 m parça: virajda karşı koldan kot alınmaz
    for c0 in range(0, n, CH):
        c1 = min(n, c0 + CH + 8)
        if c1 - c0 < 3:
            continue
        _corridor_chunk(xy[c0:c1], prof[c0:c1], hw, crown, layer, skip, kc, kf, B, write_surface, shoulder,
                        first=(c0 == 0), last=(c1 == n))
    return xy, prof


# ------------------------------------------------------------------ 2) nehir
log("nehir")
rxy, rss = resample(PLAN["river"]["points"], 0.5)
inside = (rxy[:, 0] > -20) & (rxy[:, 1] > -20)
zr = ndi.gaussian_filter1d(sample(Z, rxy), 30)
water_lvl = np.minimum.accumulate(zr - 1.6)                         # su yüzü: yukarıdan aşağı monoton
for i in range(1, len(water_lvl)):
    water_lvl[i] = min(water_lvl[i], water_lvl[i - 1] - 0.0015 * 0.5)
bed = water_lvl - 1.1
town = rxy[:, 1] > 125                                              # setler sadece kasaba kesiminde
crest = np.where(town, np.maximum(water_lvl + 2.4, zr + 0.9), water_lvl + 1.3)
crest = ndi.gaussian_filter1d(crest, 20)
bb = bbox_of(rxy, 24)
x0, x1, y0, y1 = bb
lab = np.zeros((y1 - y0, x1 - x0), np.int32)
lx = np.clip((rxy[:, 0] / CELL).astype(int) - x0, 0, x1 - x0 - 1)
ly = np.clip((rxy[:, 1] / CELL).astype(int) - y0, 0, y1 - y0 - 1)
lab[ly, lx] = np.arange(len(rxy)) + 1
dist, (iy_, ix_) = ndi.distance_transform_edt(lab == 0, return_indices=True)
d = dist * CELL
sid = lab[iy_, ix_] - 1
B_, C_, T_ = bed[sid], crest[sid], town[sid]
sub = Z[y0:y1, x0:x1]
W2 = PLAN["river"]["width_m"] / 2                                    # 7 m
prof = np.where(d <= W2 - 2.0, B_, np.where(d <= W2, B_ + (d - (W2 - 2.0)) / 2.0 * (C_ - 0.7 - B_),     # taş şev
               np.where(d <= W2 + 0.7, C_ - 0.7 + (d - W2) / 0.7 * 0.7,
                        np.where(d <= W2 + 3.7, C_, C_ - (d - W2 - 3.7) * 0.5))))                       # kret 3 m, iç şev 1:2
chan = d <= W2
sub[chan] = prof[chan]
lev = (d > W2) & (d <= W2 + 14)
sub[lev & T_] = np.maximum(sub, prof)[lev & T_]
sub[lev & ~T_] = np.minimum(sub, np.maximum(prof, sub - 0.0))[lev & ~T_]
water[y0:y1, x0:x1] |= chan
lock[y0:y1, x0:x1] |= chan
LAYER["riverbed"][y0:y1, x0:x1][d <= W2 + 0.7] = 1.0
RIVER_MASK = water.copy()
LEVEE_CREST = crest

# ------------------------------------------------------------------ 3) demiryolu (%0.8)
log("demiryolu")
rail_pts = [(-5, PLAN["rail_y"]), (446, PLAN["rail_y"])]
rail_xy, rail_prof = corridor(rail_pts, 3.0, 0.03, 0.008, 40, "gravel", skip=RIVER_MASK, kf=0.667, kc=1.0, B=12)
# peron: ray üstü + 1.1 m (araç dışı yüksek peron, Japon yerel hat standardı ~1.1 m)
for a in PLAN["areas"]:
    if a["type"] == "platform":
        x0r, y0r, x1r, y1r = a["rect"]
        sl = (slice(int(y0r / CELL), int(y1r / CELL) + 1), slice(int(x0r / CELL), int(x1r / CELL) + 1))
        Z[sl] = float(np.interp((x0r + x1r) / 2, rail_xy[:, 0], rail_prof)) + 1.1
        lock[sl] = True
        LAYER["gravel"][sl] = 1

# ------------------------------------------------------------------ 4) mabet terası ve taş basamaklar
log("mabet")
roads = {r["name"]: r for r in PLAN["roads"]}
sando_end = roads["Sando"]["points"][-1]
st = next(a for a in PLAN["areas"] if a["type"] == "steps")["rect"]
foot = _foot
TERRACE = _T
RISE = TERRACE - foot
pad_rect = [kd[0] - 1, kd[1] - 1, kd[2] + 1, kd[3] + 1]


def pad_poly(poly, h, B=10.0, kc=1.0, kf=0.667, layer=None, lock_it=True, ignore_lock=False, respect=None):
    xy = np.asarray(poly, float)
    bb = bbox_of(xy, B + 1)
    x0, x1, y0, y1 = bb
    m = poly_local(poly, bb)
    if not m.any():
        return
    sub = Z[y0:y1, x0:x1]
    lk = lock[y0:y1, x0:x1]
    inner = m & ~lk if not ignore_lock else m & ~respect[y0:y1, x0:x1]
    sub[inner] = h
    if layer:
        LAYER[layer][y0:y1, x0:x1][inner] = 1
    d = ndi.distance_transform_edt(~m) * CELL
    apr = (d > 0) & (d <= B) & ~lk
    sub[apr] = (h + np.clip(sub - h, -kf * d, kc * d))[apr]
    if lock_it:
        lk |= inner


def rect_poly(r):
    return [(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])]


# iki kademeli avlu: üst kademe honden (+2.4 m, taş duvar), alt kademe haiden/shamusho/festival deposu
_hd = next(b for b in PLAN["buildings"] if b["type"] == "haiden")
y_split = _hd["y"] - _hd["d"] / 2 - 1.5
pad_poly(rect_poly([pad_rect[0] + 4, pad_rect[1], pad_rect[2] - 4, y_split]), TERRACE + 2.4, B=20, kc=1.5, kf=0.55, layer="gravel")
pad_poly(rect_poly([pad_rect[0], y_split, pad_rect[2], pad_rect[3]]), TERRACE, B=20, kc=1.5, kf=0.55, layer="gravel")
# basamak rampası: 3 kol, 2 sahanlık (sahanlık = düz). Basamak geometrisi mesh ile oturur.
sx0, sy0, sx1, sy1 = st
L = sy1 - sy0
fl = [(0.0, 0.31), (0.38, 0.64), (0.71, 1.0)]
runs = sum(b - a for a, b in fl)
ys = np.arange(int(sy0 / CELL), int(sy1 / CELL) + 1)
for yi in ys:
    f = 1 - (yi * CELL - sy0) / L                                      # 0 = alt (ayak), 1 = üst
    acc = 0.0
    for a, b in fl[::-1]:
        fa, fb = 1 - b, 1 - a                                           # alttan
    # alttan yukarı kümülatif yükseklik
    fu = (sy1 - yi * CELL) / L
    up = 0.0
    for a, b in [(1 - b_, 1 - a_) for a_, b_ in fl[::-1]]:
        up += np.clip(fu - a, 0, b - a)
    h = foot + RISE * up / runs
    Z[yi, int(sx0 / CELL):int(sx1 / CELL) + 1] = h
    lock[yi, int(sx0 / CELL):int(sx1 / CELL) + 1] = True
    LAYER["gravel"][yi, int(sx0 / CELL):int(sx1 / CELL) + 1] = 1
# basamak yanları: doğal yamaçla şevli geçiş
pad_poly(rect_poly([sx0, sy0, sx1, sy1]), None, B=0) if False else None
xs_c = np.array([(sx0 + sx1) / 2] * 2)
STEPS = dict(foot_m=round(foot, 3), top_m=round(TERRACE, 3), upper_terrace_m=round(TERRACE + 2.4, 3), rise_m=round(RISE, 2), flights=3, landings=2, run_m=round(L, 2),
             steps_at_160mm=int(round(RISE / 0.16)))

# ------------------------------------------------------------------ 5) yollar
log("yollar")
KIND = {   # yarı genişlik eki yok; (bombe, eğim sınırı, profil yumuşatma m, katman)
    "main": (0.02, 0.06, 14, "asphalt"), "road": (0.02, 0.08, 10, "asphalt"), "lane": (0.02, 0.10, 8, "asphalt"),
    "alley": (0.015, 0.12, 5, "asphalt"), "dirt": (0.03, 0.12, 6, "dirt"), "trail": (0.02, 0.13, 5, "dirt"),
    "post": (0.02, 0.18, 4, "dirt"), "bridge": (0.02, 0.06, 0, "asphalt"),
}
GRADE_OVERRIDE = {"Sakamichi": 0.15, "Mountain Road": 0.10, "Forest link": 0.14, "Forestry road": 0.13}
ORDER = ["main", "road", "bridge", "lane", "alley", "dirt", "trail", "post"]
road_profiles = {}
bridge = roads.get("Kaido bridge")


def bridge_profile(ss, zc, pins):
    # köprü: iki ayak arası doğrusal (tabliye mesh); ayaklar mevcut yol/set kotuna sabit
    a = zc[0] if pins[0] else zc[0]
    b = zc[-1] if pins[-1] else zc[-1]
    deck = np.maximum(np.interp(ss, [ss[0], ss[-1]], [a, b]), LEVEE_CREST.max() * 0 + np.interp(ss, [ss[0], ss[-1]], [a, b]))
    return deck


# Old Kaido köprüden önce: köprü ayaklarını set kretine oturtacak şekilde Old Kaido profiline köprüyü dahil et
STEPS_MASK = np.zeros((NY, NX), bool)
STEPS_MASK[int(st[1] / CELL):int(st[3] / CELL) + 1, int(st[0] / CELL):int(st[2] / CELL) + 1] = True
KEIDAI_MASK = np.zeros((NY, NX), bool)
KEIDAI_MASK[int((kd[1] - 1) / CELL):int((kd[3] + 1) / CELL) + 1, int((kd[0] - 1) / CELL):int((kd[2] + 1) / CELL) + 1] = True
SKIP_R = RIVER_MASK | STEPS_MASK | KEIDAI_MASK
def _touches(r, others):
    """Bu yolun kaç ucu aynı türden başka bir yolun üzerine bağlanıyor (kol yollar gövdeden sonra işlenir)."""
    n = 0
    for e in (r["points"][0], r["points"][-1]):
        for o in others:
            if o is r:
                continue
            q = np.asarray(o["points"])
            if np.min(np.hypot(q[:, 0] - e[0], q[:, 1] - e[1])) < o["width_m"] + 2:
                n += 1
                break
    return n


for kind in ORDER:
    same = [r for r in PLAN["roads"] if r["kind"] == kind]
    for r in sorted(same, key=lambda r: (_touches(r, same), -len(r["points"]))):
        crown, g, sig, layer = KIND[kind]
        g = GRADE_OVERRIDE.get(r["name"], g)
        hw = r["width_m"] / 2
        if kind == "bridge":
            xy, prof = corridor(r["points"], hw, crown, g, sig, layer, skip=SKIP_R, prof_override=bridge_profile)
        else:
            if kind in ("trail", "post", "dirt") and r["name"] not in ("West Bank Lane", "SE farm road", "Farm road E", "Farm road S", "Farm road SW"):
                xy, prof = corridor(r["points"], hw, 0.0 if kind != "dirt" else 0.02, g, 3.0, layer, skip=SKIP_R, kc=1.6, kf=0.9, B=5, shoulder=0.2)
                road_profiles[r["name"]] = (xy, prof, hw, crown, g)
                continue
            xy, prof = corridor(r["points"], hw, crown, g, sig, layer, skip=SKIP_R,
                                kf=0.667 if kind in ("main", "road", "lane") else 0.8, kc=1.5 if r["name"] in ("Mountain Road", "Sakamichi", "Forestry road", "Forest link", "Old postal path") else 1.0,
                                B=18 if kind in ("main", "road") else 10)
        road_profiles[r["name"]] = (xy, prof, hw, crown, g)

# ------------------------------------------------------------------ 6) sulama kanalları
log("kanallar")
CH = []
for c in PLAN["channels"]:
    xy, ss = resample(c["points"], 0.5)
    zc = sample(Z, xy)
    wl = ndi.gaussian_filter1d(zc, 8) - 0.35
    wl = np.minimum.accumulate(wl)
    for i in range(1, len(wl)):
        wl[i] = min(wl[i], wl[i - 1] - 0.001 * 0.5)
    bedc = wl - 0.45
    hw = c["width_m"] / 2
    bb = bbox_of(xy, hw + 3)
    x0, x1, y0, y1 = bb
    lab = np.zeros((y1 - y0, x1 - x0), np.int32)
    lab[np.clip((xy[:, 1] / CELL).astype(int) - y0, 0, y1 - y0 - 1), np.clip((xy[:, 0] / CELL).astype(int) - x0, 0, x1 - x0 - 1)] = np.arange(len(xy)) + 1
    dist, (iy_, ix_) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    d = dist * CELL
    sid = lab[iy_, ix_] - 1
    sub = Z[y0:y1, x0:x1]
    lk = lock[y0:y1, x0:x1]
    top = sub.copy()
    prof = np.where(d <= hw - 0.15, bedc[sid], bedc[sid] + (d - hw + 0.15) / 0.3 * (top - bedc[sid]))       # beton U-kanal: dik duvar
    m = (d <= hw + 0.15) & ~lk
    sub[m] = np.minimum(sub, prof)[m]
    water[y0:y1, x0:x1] |= (d <= hw) & ~lk
    lk |= (d <= hw + 0.15) & ~lk
    LAYER["riverbed"][y0:y1, x0:x1][(d <= hw + 0.15)] = np.maximum(LAYER["riverbed"][y0:y1, x0:x1][(d <= hw + 0.15)], 1)
    CH.append(dict(name=c["name"], water_start=round(float(wl[0]), 3), water_end=round(float(wl[-1]), 3), length=round(float(ss[-1]), 1)))

# ------------------------------------------------------------------ 7) alanlar, parseller, bina pedleri
log("alan / parsel / bina pedleri")


def front_height(poly):
    """Parsel ön kenarının yol tarafındaki kot (ön kenar ortası, 1.2 m dışarı)."""
    (ax, ay), (bx, by), (cx, cy), (dx, dy) = poly
    mx, my = (ax + bx) / 2, (ay + by) / 2
    gx, gy = (cx + dx) / 2, (cy + dy) / 2
    L = math.hypot(gx - mx, gy - my) or 1
    ox, oy = mx - (gx - mx) / L * 1.4, my - (gy - my) / L * 1.4
    return float(sample(Z, np.array([[ox, oy]]))[0])


for a in PLAN["areas"]:
    if a["type"] in ("keidai", "steps", "platform"):
        continue
    poly = rect_poly(a["rect"])
    xy = np.asarray(poly)
    h = float(np.median(sample(Z, np.vstack([xy, xy.mean(0, keepdims=True)]))))
    lay = {"parking": "asphalt", "plaza": "asphalt", "schoolyard": "gravel", "park": "gravel"}.get(a["type"])
    pad_poly(poly, h + 0.05, B=8, layer=lay)
PARCEL_PADS = {}
BY_PARCEL = {}
for b in PLAN["buildings"]:
    if b.get("parcel") is not None:
        BY_PARCEL.setdefault(b["parcel"], []).append(b)


def bpoly(b, m):
    c, s_ = math.cos(b["ang"]), math.sin(b["ang"])
    w, d_ = b["w"] + 2 * m, b["d"] + 2 * m
    return [(b["x"] + u * c - v * s_, b["y"] + u * s_ + v * c) for u, v in ((-w / 2, -d_ / 2), (w / 2, -d_ / 2), (w / 2, d_ / 2), (-w / 2, d_ / 2))]


for p in PLAN["parcels"]:
    poly = p["poly"]
    xy = np.asarray(poly)
    med = float(np.median(sample(Z, np.vstack([xy, xy.mean(0, keepdims=True)]))))
    if p["zone"] in ("mtnroad", "mountain"):
        # yamaç evi: parselin tamamı değil, yalnız ev + 2.5 m avlu düzlenir (taş duvarlı küçük teras, ishigaki)
        PARCEL_PADS[p["id"]] = None
        for b in BY_PARCEL.get(p["id"], []):
            if b["type"] in ("porch",):
                continue
            pp = bpoly(b, 2.5 if b["type"] not in ("carport", "shed", "kura") else 1.0)
            pxy = np.asarray(pp)
            h = float(np.median(sample(Z, np.vstack([pxy, [[b["x"], b["y"]]]]))))
            pad_poly(pp, h, B=10, kc=1.5, kf=0.7, layer="garden")
        continue
    if p["zone"] in ("farm", "rural"):
        h, B, kc, kf = med, 10, 1.0, 0.6
    else:
        h, B, kc, kf = front_height(poly) + 0.12, 6, 1.2, 0.8
    PARCEL_PADS[p["id"]] = h
    pad_poly(poly, h, B=B, kc=kc, kf=kf, layer="garden")
PAD_REPORT = []
for b in PLAN["buildings"]:
    if b.get("parcel") is not None and b["parcel"] in PARCEL_PADS:  # parsel pedinde (yamaç evleri ayrıca)
        continue
    if b["type"] in ("torii", "tunnel", "radio_mast", "lookout"):
        continue
    c, s = math.cos(b["ang"]), math.sin(b["ang"])
    w, d_ = b["w"] + 2.0, b["d"] + 2.0
    poly = [(b["x"] + u * c - v * s, b["y"] + u * s + v * c) for u, v in ((-w / 2, -d_ / 2), (w / 2, -d_ / 2), (w / 2, d_ / 2), (-w / 2, d_ / 2))]
    xy = np.asarray(poly)
    if b["type"] in ("honden", "haiden", "shamusho", "festival_store", "temizuya"):
        continue                                                     # teras zaten düz
    h = float(np.median(sample(Z, np.vstack([xy, [[b["x"], b["y"]]]]))))
    roadsurf = (LAYER["asphalt"] > 0) | (LAYER["dirt"] > 0) | water
    pad_poly(poly, h, B=8, kc=1.2, kf=0.8, ignore_lock=True, respect=roadsurf)

# ------------------------------------------------------------------ 8) çeltik terasları ve bostan
log("çeltik terasları")
PADDY_CELLS = []
for f in PLAN["fields"]:
    poly = f["poly"]
    xy = np.asarray(poly)
    bb = bbox_of(xy, 0)
    x0, x1, y0, y1 = bb
    fm = poly_local(poly, bb) & ~lock[y0:y1, x0:x1] & ~water[y0:y1, x0:x1]
    if f["kind"] == "veg":
        sub = Z[y0:y1, x0:x1]
        sm = ndi.gaussian_filter(sub, 6)
        sub[fm] = sm[fm]
        LAYER["garden"][y0:y1, x0:x1][fm] = 1
        continue
    cw, ch = 16.0, 11.0
    for cy in np.arange(xy[:, 1].min(), xy[:, 1].max(), ch):
        for cx in np.arange(xy[:, 0].min(), xy[:, 0].max(), cw):
            cx0, cy0 = int(cx / CELL) - x0, int(cy / CELL) - y0
            cx1, cy1 = int((cx + cw) / CELL) - x0, int((cy + ch) / CELL) - y0
            cx0, cy0 = max(cx0, 0), max(cy0, 0)
            cell = np.zeros_like(fm)
            cell[cy0:cy1, cx0:cx1] = True
            cell &= fm
            cell = ndi.binary_erosion(cell, iterations=1)
            if cell.sum() < 300:
                continue
            sub = Z[y0:y1, x0:x1]
            h = float(np.round(np.median(sub[cell]) / 0.05) * 0.05)       # 5 cm kademeli teras kotu
            sub[cell] = h - 0.10                                          # tarla yüzü (su tutar)
            bund = cell & ~ndi.binary_erosion(cell, iterations=2)          # 0.5 m aze sırtı
            sub[bund] = h + 0.15
            lock[y0:y1, x0:x1] |= cell
            LAYER["paddy"][y0:y1, x0:x1][cell & ~bund] = 1
            LAYER["dirt"][y0:y1, x0:x1][bund] = 1
            PADDY_CELLS.append(dict(x=round(cx, 1), y=round(cy, 1), w=cw, h=ch, z=round(h - 0.1, 3)))

# ------------------------------------------------------------------ yol yüzeylerini yeniden bas (düşük öncelik önce, ana yol en son kazanır)
for (x0, x1, y0, y1), m, val in SURF[::-1]:
    Z[y0:y1, x0:x1][m] = val[m]

# ------------------------------------------------------------------ doğal alanlarda hafif yumuşatma (kilitli olmayan)
log("son yumuşatma")
free = ~lock & ~water
near_lock = ndi.binary_dilation(lock | water, iterations=1)
sm = ndi.gaussian_filter(Z, 2.0)
Z[free & ~near_lock] = sm[free & ~near_lock]

# ------------------------------------------------------------------ QA
log("QA")
qa = {"roads": {}, "worst": {}}
worst_g = worst_x = 0
for name, (xy, prof, hw, crown, g) in road_profiles.items():
    zc = sample(Z, xy)
    ds = 0.25
    k = max(1, int(2.0 / ds))                              # 2 m taban üzerinden eğim
    gr = np.abs(zc[k:] - zc[:-k]) / (k * ds) if len(zc) > k else np.array([0.0])
    if name == "Kaido bridge":
        continue
    excl = ndi.binary_dilation(RIVER_MASK | STEPS_MASK, iterations=12)
    on_river = excl[np.clip((xy[:, 1] / CELL).astype(int), 0, NY - 1), np.clip((xy[:, 0] / CELL).astype(int), 0, NX - 1)]
    on_river |= (xy[:, 0] < 1) | (xy[:, 0] > 439) | (xy[:, 1] < 1) | (xy[:, 1] > 344)
    gr_land = gr[~on_river[k:]] if len(gr) and (~on_river[k:]).any() else np.array([0.0])
    # enine: merkez ile ±0.8*hw arası
    t = np.gradient(xy, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    zl = sample(Z, xy + nrm * hw * 0.8)
    zr = sample(Z, xy - nrm * hw * 0.8)
    cs = np.maximum(np.abs(zc - zl), np.abs(zc - zr)) / (hw * 0.8)
    p95 = float(np.percentile(gr_land, 99)) if len(gr_land) else 0
    qa["roads"][name] = dict(max_grade_pct=round(float(gr_land.max()) * 100, 2), p99_grade_pct=round(p95 * 100, 2),
                             limit_pct=round(g * 100, 1), cross_slope_median_pct=round(float(np.median(cs)) * 100, 2),
                             z_min=round(float(zc.min()), 2), z_max=round(float(zc.max()), 2), length_m=round(len(zc) * ds, 1))
    worst_g = max(worst_g, gr_land.max() - g)
over = {n: v for n, v in qa["roads"].items() if v["p99_grade_pct"] > v["limit_pct"] + 0.5}
qa["roads_over_grade_limit_p99"] = over
flat = []
for b in PLAN["buildings"]:
    if b["type"] in ("torii", "tunnel", "radio_mast", "lookout", "porch", "carport", "open_shed"):
        continue
    c, s = math.cos(b["ang"]), math.sin(b["ang"])
    u, v = np.meshgrid(np.linspace(-b["w"] / 2 + 0.3, b["w"] / 2 - 0.3, 6), np.linspace(-b["d"] / 2 + 0.3, b["d"] / 2 - 0.3, 6))
    pts = np.stack([(b["x"] + u * c - v * s).ravel(), (b["y"] + u * s + v * c).ravel()], 1)
    zz = sample(Z, pts)
    flat.append(float(zz.max() - zz.min()))
flat = np.array(flat)
qa["building_pad_flatness_m"] = dict(median=round(float(np.median(flat)), 3), p95=round(float(np.percentile(flat, 95)), 3),
                                     max=round(float(flat.max()), 3), over_5cm=int((flat > 0.05).sum()), count=len(flat))
qa["river"] = dict(bed_monotonic=bool(np.all(np.diff(bed) <= 1e-9)), water_top=round(float(water_lvl[inside].max()), 2),
                   water_bottom=round(float(water_lvl[inside].min()), 2), levee_crest_town_min=round(float(crest[town].min()), 2))
qa["channels"] = CH
qa["shrine"] = STEPS
qa["paddy_cells"] = len(PADDY_CELLS)
qa["z_range_m"] = [round(float(Z.min()), 2), round(float(Z.max()), 2)]
# gerçek Kiso referansı ile karşılaştırma: ormanlık yamaç eğimi ve 440x345 m pencere rölyefi
_gy, _gx = np.gradient(ndi.zoom(Z, CELL / RES, order=1), RES)
_sl = np.degrees(np.arctan(np.hypot(_gx, _gy)))
_fm = ndi.zoom(((mtn_f > 0.6) & ~ndi.binary_dilation(lock, iterations=48)).astype(float), CELL / RES, order=0) > 0.5
_rgy, _rgx = np.gradient(DEM, RES)
_rsl = np.degrees(np.arctan(np.hypot(_rgx, _rgy)))
qa["reference_compare"] = dict(
    ours_forest_slope_deg=dict(p50=round(float(np.percentile(_sl[_fm], 50)), 1), p90=round(float(np.percentile(_sl[_fm], 90)), 1)),
    kiso_narai_slope_deg=dict(p50=round(float(np.percentile(_rsl, 50)), 1), p90=round(float(np.percentile(_rsl, 90)), 1)),
    ours_relief_m=round(float(np.ptp(Z)), 1), kiso_relief_440x345_p10_p50=[124, 223], source=REF_INFO)

# ------------------------------------------------------------------ çıktı: 16-bit heightmap + Unreal ayarları
zmin, zmax = float(Z.min()), float(Z.max())
span = zmax - zmin
zscale = math.ceil(span / 512.0 * 100 * 1.04 * 10) / 10                # Z ölçeği (Unreal: 100 = 512 m aralık)
zmid = (zmin + zmax) / 2
u16 = np.clip(np.round(32768 + (Z - zmid) * 128 * 100 / zscale), 0, 65535).astype(np.uint16)
Image.fromarray(u16).save(os.path.join(OUT, "mizunagi_height.png"))
u16.astype("<u2").tofile(os.path.join(OUT, "mizunagi_height.r16"))
precision_mm = zscale / 100 / 128 * 1000

# katmanlar: öncelikli maske + kenar yumuşatma, toplam 255
log("katmanlar")
forest_m = np.zeros((NY, NX), np.float32)
grass = np.ones((NY, NX), np.float32)
L_ = {}
L_["riverbed"] = LAYER["riverbed"]
L_["asphalt"] = LAYER["asphalt"] * (1 - L_["riverbed"])
L_["gravel"] = LAYER["gravel"] * (1 - np.maximum(L_["riverbed"], L_["asphalt"]))
L_["dirt"] = LAYER["dirt"] * (1 - np.maximum.reduce([L_["riverbed"], L_["asphalt"], L_["gravel"]]))
taken = np.maximum.reduce([L_["riverbed"], L_["asphalt"], L_["gravel"], L_["dirt"]])
L_["paddy"] = LAYER["paddy"] * (1 - taken)
taken = np.maximum(taken, L_["paddy"])
L_["garden"] = LAYER["garden"] * (1 - taken)
taken = np.maximum(taken, L_["garden"])
slope = np.hypot(*np.gradient(Z, CELL))
L_["forest"] = (np.clip((mtn_f - 0.12) * 5, 0, 1) * (1 - taken)).astype(np.float32)
L_["rock"] = (np.clip((slope - 0.9) * 2.5, 0, 1) * L_["forest"]).astype(np.float32)
L_["forest"] *= (1 - L_["rock"])
taken = np.maximum.reduce([taken, L_["forest"], L_["rock"]])
L_["grass"] = 1 - taken
for k in L_:
    L_[k] = ndi.gaussian_filter(L_[k], 0.8)
tot = sum(L_.values()) + 1e-6
for k, v in L_.items():
    Image.fromarray(np.round(v / tot * 255).astype(np.uint8)).save(os.path.join(OUT, f"layer_{k}.png"))

# önizleme: tepe gölgesi + hipsometrik + kontur 1 m
log("önizleme")
gy, gx = np.gradient(Z, CELL)
az, alt = math.radians(315), math.radians(40)
# normal ~ (-gx, -gy, 1); ışığa doğru birim vektör (doğu, güney, yukarı) = (sin az, -cos az, ...) * cos alt
lx, ly = math.sin(az) * math.cos(alt), -math.cos(az) * math.cos(alt)
hs = np.clip((-gx * lx - gy * ly + math.sin(alt)) / np.sqrt(1 + gx ** 2 + gy ** 2), 0, 1)
t = (Z - zmin) / span
tint = np.stack([0.55 + 0.35 * t, 0.62 + 0.25 * t, 0.48 + 0.3 * t], -1)
rgb = tint * (0.35 + 0.75 * hs[..., None])
rgb[water] = rgb[water] * 0.5 + np.array([0.35, 0.5, 0.65]) * 0.5
def _cont(iv):
    q = np.floor(Z / iv)
    return (q != np.floor(ndi.shift(Z, (0, 1), order=0, mode="nearest") / iv)) | (q != np.floor(ndi.shift(Z, (1, 0), order=0, mode="nearest") / iv))


rgb[_cont(5.0) & ~lock] *= 0.82
rgb[ndi.binary_dilation(_cont(25.0)) & ~lock] *= 0.7
asp = L_["asphalt"] / tot > 0.5
rgb[asp] = rgb[asp] * 0.55 + 0.45 * np.array([0.92, 0.9, 0.86])
img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
img.save(os.path.join(OUT, "preview_full.png"))
img.crop((int(95 / CELL), int(135 / CELL), int(330 / CELL), int(320 / CELL))).save(os.path.join(OUT, "preview_town.png"))
# gerçek Kiso ile yan yana: aynı ölçek (3.87 m/px), aynı ışık
def _hs(D, res):
    gy_, gx_ = np.gradient(D, res)
    lx_, ly_ = math.sin(az) * math.cos(alt), -math.cos(az) * math.cos(alt)
    return np.clip((-gx_ * lx_ - gy_ * ly_ + math.sin(alt)) / np.sqrt(1 + gx_ ** 2 + gy_ ** 2), 0, 1)


ours = ndi.zoom(Z, CELL / RES, order=1)
ny_, nx_ = ours.shape
i0, j0 = REF_INFO["window_north"][0], REF_INFO["window_north"][1]
refw = DEM[i0:i0 + ny_, j0:j0 + nx_]
cmp_ = np.hstack([_hs(ours, RES), np.ones((ny_, 4)), _hs(refw, RES)[:ny_, :nx_]])
Image.fromarray((cmp_ * 255).astype(np.uint8)).resize((cmp_.shape[1] * 3, ny_ * 3), Image.BICUBIC).save(os.path.join(OUT, "preview_reference_compare.png"))
hs_img = Image.fromarray((hs * 255).astype(np.uint8))
hs_img.save(os.path.join(OUT, "preview_hillshade.png"))

meta = dict(
    file="mizunagi_height.png (16-bit) / mizunagi_height.r16", size_px=[NX, NY], cell_m=CELL, extent_m=[NX * CELL - CELL, NY * CELL - CELL],
    unreal=dict(section_size="63x63 quads", sections_per_component="1x1", components=[28, 22], overall_resolution=[NX, NY],
                scale=[CELL * 100, CELL * 100, zscale], location_z_cm=round(zmid * 100, 1),
                note="Landscape Location Z = location_z_cm; harita (0,0) = sol üst = kuzeybatı; +X doğu, +Y güney"),
    height_range_m=[round(zmin, 3), round(zmax, 3)], precision_mm=round(precision_mm, 3),
    layers=sorted(f"layer_{k}.png" for k in L_), terrace_z_m=round(TERRACE, 3), qa=qa,
    river_water_level=dict(start=round(float(water_lvl[inside][0]), 3), end=round(float(water_lvl[inside][-1]), 3)),
    bridge_note="Köprü ve demiryolu köprüsü altı nehir yatağıdır; tabliye mesh olarak yerleştirilmeli (yol profili JSON'da).",
    road_profiles={n: dict(points=[[round(float(x), 2), round(float(y), 2)] for x, y in xy[::8]],
                           z=[round(float(z), 3) for z in prof[::8]]) for n, (xy, prof, *_r) in road_profiles.items()},
)
json.dump(meta, open(os.path.join(OUT, "height_meta.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: meta[k] for k in ("size_px", "height_range_m", "precision_mm", "unreal")}, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in qa.items() if k != "roads"}, ensure_ascii=False, indent=1))
log("tamam ->", OUT)
