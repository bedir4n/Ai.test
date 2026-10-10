# Mizunagi-chō şehir planı

`python3 generate.py` → `out/mizunagi_plan.png` + `out/mizunagi_plan.json`

- Ölçek: gerçek metre, 440 × 352 m, 1 px = 0,4 m (PNG 2,5 px/m).
- Sıra: bölgeler (orman/tarla) → yollar → özel yapılar (tapınak, istasyon, okul, market, otopark…) → sokak boyunca parseller → lamba/direk/otomat → QA.
- Bina yola/nehre taşamaz; üretim sırasında çarpışma testi yapılır, QA raporu JSON'da (`qa`).
- Boş alanlar orman (Sugi, kuzey/batı yamaç) ya da ekin/çeltik tarlasıdır (güneydoğu, batı yaka, doğu).
- JSON: binalar (x,y,w,d,açı,tür,kat), yollar, alanlar, lamba/direk/otomat konumları — Unreal'e aktarmak için.
