# Mizunagi-machi yerleşim planı (1991)

`python3 generate.py` → `out/mizunagi_plan.png` (harita + ayrı açıklama kutusu) ve `out/mizunagi_plan.json` (Unreal'e aktarım verisi).

- **Ölçek:** Gerçek metre kullanılıyor, harita 440 × 345 m. Yol ağı referans çizimden alınıyor (1 ref px = 0,2366 m).
- **Parsel tabanlı:** Her konutun bir parseli var. Parselde duvar (taş, beton blok, ahşap çit, çit bitkisi), kapı, posta kutusu, genkan, ön ve arka bahçe, kura/depo, carport ya da park cebi bulunuyor. Arka sıradaki bayrak parseller yola dar bir geçitle bağlanıyor, posta kutuları sokak ağzında.
- **Yoğunluk merkezden dışa azalıyor:** Furumachi → Old Kaido → Shotengai → Core → East Road → çiftlik kuşağı → dağ.
- **Kültür:** Yalnızca Şinto mabedi var: torii, taş basamak (3 kol, 2 sahanlık), temizuya, haiden, honden, shamusho, festival deposu ve kutsal koruluk. Ticaret: JA kooperatifi, kısa bir shotengai ve tek bir modern market.
- **Arazi:** Sugi plantasyonu düzenli sıralarla ve kesim açıklıklarıyla gösteriliyor, doğal karışık ormandan ayrı. Çeltik tarlaları savak, kanal ve tarla içi hendeklerle nehre bağlı.
- **Oyun verisi:** JSON'da `delivery` alanı teslimat adresinin numarasını, `enterable` alanı iç mekânı girilebilir yapıları işaretliyor.
- **QA:** Çıktının `qa` alanında. Yola taşan, suya taşan veya birbiriyle çakışan bina ve yola taşan parsel sayısı 0 olmalı.
