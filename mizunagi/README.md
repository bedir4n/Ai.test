# Mizunagi-machi yerleşim planı (1991)

`python3 generate.py` → `out/mizunagi_plan.png` (harita + ayrı açıklama kutusu) ve `out/mizunagi_plan.json` (Unreal'e aktarım verisi).

- **Ölçek:** Gerçek metre kullanılıyor, harita 440 × 345 m. Yol ağı referans çizimden alınıyor (1 ref px = 0,2366 m).
- **Parsel tabanlı:** Her konutun bir parseli var. Parselde duvar (taş, beton blok, ahşap çit, çit bitkisi), kapı, posta kutusu, genkan, ön ve arka bahçe, kura/depo, carport ya da park cebi bulunuyor. Arka sıradaki bayrak parseller yola dar bir geçitle bağlanıyor, posta kutuları sokak ağzında.
- **Yoğunluk merkezden dışa azalıyor:** Furumachi → Old Kaido → Shotengai → Core → East Road → çiftlik kuşağı → dağ.
- **Kültür:** Yalnızca Şinto mabedi var: torii, taş basamak (3 kol, 2 sahanlık), temizuya, haiden, honden, shamusho, festival deposu ve kutsal koruluk. Ticaret: JA kooperatifi, kısa bir shotengai ve tek bir modern market.
- **Arazi:** Sugi plantasyonu düzenli sıralarla ve kesim açıklıklarıyla gösteriliyor, doğal karışık ormandan ayrı. Çeltik tarlaları savak, kanal ve tarla içi hendeklerle nehre bağlı.
- **Oyun verisi:** JSON'da `delivery` alanı teslimat adresinin numarasını, `enterable` alanı iç mekânı girilebilir yapıları işaretliyor.
- **QA:** Çıktının `qa` alanında. Yola taşan, suya taşan veya birbiriyle çakışan bina ve yola taşan parsel sayısı 0 olmalı.

## Yükseklik haritası (Unreal Landscape)

`python3 heightmap.py` komutu planı (`out/mizunagi_plan.json`) okur ve `out/height/` klasörüne şunları yazar:

| Dosya | İçerik |
|---|---|
| `mizunagi_height.png` / `.r16` | 16-bit yükseklik, 1765 × 1387 (28 × 22 bileşen, 63×63 quad, 1 bölüm) |
| `layer_*.png` | 8-bit katman ağırlıkları: asfalt, toprak yol, çakıl, çeltik, su yatağı, bahçe, orman, kaya, çim (toplamı 255) |
| `height_meta.json` | Import ölçeği, Location Z, QA raporu, yol profilleri (köprü tabliyeleri için) |
| `preview_*.png` | Gölgelendirilmiş önizlemeler |

**Import ayarları:** Scale X/Y = 25, Z ve Location Z değerleri `height_meta.json` içindeki `unreal` alanında. Haritanın (0,0) noktası kuzeybatı köşe; +X doğu, +Y güney.

**Doğal arazi:** Dağlar tasarım kotunda bir makro yüzeyle başlıyor: net dağ eteği, gövdede ~30°. Bu yüzeye iki ölçekli, damla tabanlı hidrolik erozyon (`erosion.py`, Gaea/World Machine yöntemi) uygulanıyor:
- 2 m ızgarada ana dereler oluşuyor, aralarında gerçek ölçekte 15–25 m var.
- 1 m ızgarada yamaç olukları ekleniyor.
- 40°'yi aşan yamaçlar taşla kayıyor; tortunun dağ eteğinde yığılması sınırlı.

Eğim dağılımı gerçek Kiso (Narai) verisiyle karşılaştırılıyor: `ref/kiso_reference.npz`, QA'da `reference_compare` alanı.

**Neler var:**
- **Yollar:** Boyuna profil araziden yumuşatılıyor ve eğim sınırına uyduruluyor (ana yol %6, yol %8, sokak %10, roji %12, Sakamichi %15). Kavşaklar önce yapılan yola sabitleniyor. Enine %2 bombe ve 0,4 m banket var; şevler kesmede 1:1, dolguda 1:1,5. Kotlar yol eksenine izdüşümle hesaplandığı için kademe oluşmuyor. Virajlarda karşı koldan kot alınmaması için yol 15 m'lik parçalarla işleniyor.
- **Nehir:** Yatak yukarıdan aşağı monoton iniyor. Taş şev, 3 m genişliğinde set kreti ve 1:2 iç şev var. Dağ kesiminde vadi V şeklinde oyuluyor.
- **Demiryolu:** En fazla %0,8 eğim, balastlı yatak, peron ray kotunun 1,1 m üstünde.
- **Mabet:** Yamaca oturan iki kademeli avlu: alt kademede haiden, 2,4 m yukarıdaki üst kademede honden; arkası taş duvarlı kazı. Basamak eteğinden alt avluya 16 m yükselti var, 3 kol ve 2 sahanlık halinde.
- **Kanallar:** Akış yönünde monoton, beton U-kanal kesiti. Yol altında menfez bırakılıyor.
- **Çeltik:** Her tarla 5 cm kademeli düz bir teras, kenarlarında 0,5 m'lik aze sırtı var.
- **Parseller:** Ped kotu ön kenardaki yol kotunun 12 cm üstünde. Yamaç evlerinde parselin tamamı değil, yalnız ev ve 2,5 m avlusu düzleniyor (taş duvarlı küçük teras, ishigaki).
- **Mesh olarak eklenmesi gerekenler:** Köprü tabliyeleri, basamaklar, istinat duvarları.
