"""Mizunagi: yeni heightmap'i Unreal'de test landscape'i olarak kur (HİÇBİR ŞEY KAYDETMEZ).

Çalıştırma: Unreal Editor -> Tools -> Execute Python Script... -> bu dosya
(Python Editor Script Plugin açık olmalı.)

Ne yapar:
  1. Sahnedeki mevcut landscape'i bulur (MI_Landscape_Master kullananı tercih eder) ve DOKUNMADAN kopyalar.
  2. mizunagi_height_rg8.png'yi doku olarak içe aktarır (16-bit yükseklik R/G kanallarına paketli).
  3. Kopyanın yüksekliğini bu dokudan yazar (render target üzerinden).
  4. MI_Landscape_Master materyalini takar, ölçek/konumu height_meta.json'a göre ayarlar.
  5. Eski landscape'i editörde GEÇİCİ olarak gizler (silmez, kaydetmez).
Level ve asset'ler kaydedilmez; beğenmezsen "Don't Save" ile kapat, her şey eski haline döner.
"""
import json
import os
import unreal

# ---- dosya yolları: depodaki mizunagi/out/height klasörünü buraya kopyala
HEIGHT_DIR = r"C:\Users\bedir\Desktop\MizunagiMap\height"
RG8_PNG = os.path.join(HEIGHT_DIR, "mizunagi_height_rg8.png")
META = os.path.join(HEIGHT_DIR, "height_meta.json")
DEST = "/Game/Mizunagi/HeightTest"
NEW_LABEL = "Mizunagi_Landscape_v3_TEST"


def log(msg):
    unreal.log_warning("[Mizunagi] " + msg)


meta = json.load(open(META, encoding="utf-8"))
z_scale = float(meta["unreal"]["scale"][2])
loc_z = float(meta["unreal"]["location_z_cm"])
extent_m = meta["extent_m"]                      # [441.0, 346.5]

eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()

# ---- 1) kaynak landscape
lands = [a for a in eas.get_all_level_actors() if isinstance(a, unreal.Landscape)]
if not lands:
    raise RuntimeError("Sahnede landscape yok. Önce boş bir landscape oluştur (Landscape Mode > New > Create).")


def mat_name(l):
    m = l.get_editor_property("landscape_material")
    return m.get_name() if m else ""


src = next((l for l in lands if "MI_Landscape_Master" in mat_name(l)), lands[0])
log("Kaynak landscape: %s (materyal: %s)" % (src.get_actor_label(), mat_name(src) or "-"))

# ---- 2) kopya (kaynak değişmez)
eas.select_nothing()
new = eas.duplicate_actor(src, world)
new.set_actor_label(NEW_LABEL)
log("Kopya oluşturuldu: " + NEW_LABEL)

# çözünürlük: sınırlar / ölçek
origin, box = new.get_actor_bounds(False)
sc = new.get_actor_scale3d()
res_x = int(round(2 * box.x / sc.x)) + 1
res_y = int(round(2 * box.y / sc.y)) + 1
log("Kopya landscape çözünürlüğü: %d x %d (heightmap 1765 x 1387 buna örneklenecek)" % (res_x, res_y))

# ---- 3) dokuyu içe aktar (kaydetmeden)
task = unreal.AssetImportTask()
task.filename = RG8_PNG
task.destination_path = DEST
task.destination_name = "T_Mizunagi_Height_RG8"
task.automated = True
task.replace_existing = True
task.save = False
unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
tex = unreal.load_asset(DEST + "/T_Mizunagi_Height_RG8")
if tex is None:
    raise RuntimeError("Doku içe aktarılamadı: " + RG8_PNG)
tex.set_editor_property("srgb", False)
tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_VECTOR_DISPLACEMENTMAP)  # sıkıştırmasız RGBA8
tex.set_editor_property("filter", unreal.TextureFilter.TF_NEAREST)        # paketli baytlar karışmasın
tex.set_editor_property("mip_gen_settings", unreal.TextureMipGenSettings.TMGS_NO_MIPMAPS)
tex.set_editor_property("never_stream", True)
tex.post_edit_change()

# ---- 4) render target'a çiz ve landscape'e yaz
rt = unreal.RenderingLibrary.create_render_target2d(world, res_x, res_y, unreal.TextureRenderTargetFormat.RTF_RGBA8,
                                                    unreal.LinearColor(0, 0, 0, 1), False)
canvas, size, ctx = unreal.RenderingLibrary.begin_draw_canvas_to_render_target(world, rt)
canvas.draw_texture(tex, unreal.Vector2D(0, 0), unreal.Vector2D(res_x, res_y), unreal.Vector2D(0, 0), unreal.Vector2D(1, 1),
                    unreal.LinearColor(1, 1, 1, 1), unreal.BlendMode.BLEND_OPAQUE, 0.0, unreal.Vector2D(0.5, 0.5))
unreal.RenderingLibrary.end_draw_canvas_to_render_target(world, ctx)
ok = new.landscape_import_heightmap_from_render_target(rt, True)      # True: yükseklik R/G kanallarında paketli
log("Yükseklik yazıldı: %s" % ok)

# ---- 5) materyal, ölçek, konum
mat = None
reg = unreal.AssetRegistryHelpers.get_asset_registry()
for ad in reg.get_assets_by_class(unreal.TopLevelAssetPath("/Script/Engine", "MaterialInstanceConstant"), True):
    if str(ad.asset_name) == "MI_Landscape_Master":
        mat = ad.get_asset()
        break
if mat:
    new.set_editor_property("landscape_material", mat)
    log("Materyal takıldı: " + mat.get_path_name())
else:
    log("MI_Landscape_Master bulunamadı; kaynak landscape'in materyali kaldı.")

sx = extent_m[0] * 100.0 / (res_x - 1)
sy = extent_m[1] * 100.0 / (res_y - 1)
new.set_actor_scale3d(unreal.Vector(sx, sy, z_scale))
loc = src.get_actor_location()
new.set_actor_location(unreal.Vector(loc.x, loc.y, loc_z), False, False)
log("Ölçek X %.2f  Y %.2f  Z %.2f, Location Z %.1f cm" % (sx, sy, z_scale, loc_z))

# ---- 6) eski landscape'i geçici gizle (silmez, kaydedilmez)
src.set_is_temporarily_hidden_in_editor(True)
eas.set_selected_level_actors([new])
log("Tamam. Eski landscape geçici olarak gizlendi. Hiçbir şey kaydedilmedi.")
if res_x < 1765:
    log("Not: kopya %dx%d çözünürlükte; tam kalite (25 cm) için 1765x1387 boyutunda landscape gerekir." % (res_x, res_y))
