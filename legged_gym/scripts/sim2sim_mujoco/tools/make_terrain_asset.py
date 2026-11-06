# ==============================================================================
# Date      : 2026-11-06
# Author    : SYH
# File      : make_terrain_asset.py
# Purpose   : 把训练侧预生成的高度图 npz（wbr_p8_demo.npz 等）转成 MuJoCo 部署资产：
#             ① 生成一份带 <hfield> 的 MJCF（替换 plane 地面）；
#             ② 生成 .npy 归一化高度数据（运行时写 m.hfield_data，绕开 MuJoCo 的
#                PNG 加载——实测 3.17 的 <hfield file=...> 对 8/16-bit PNG 都读成 0）。
#
#   坐标口径（与 legged_gym/utils/terrain.py 的 trimesh/heightfield 对齐）：
#     npz.height_field_raw : int16，shape (tot_rows, tot_cols)，实际高度 = raw * v_scale
#     ⚠ 训练侧 npz 的 row=x(length)、col=y(width)；但 MuJoCo <hfield> 的 nrow 沿 **y**、
#       ncol 沿 **x**（实测：795x570 配 size(28.5,39.75) 才碰撞正常，即 size[0]=ncol*hs/2、
#       size[1]=nrow*hs/2）。故 data 必须转置后再写 m.hfield_data，否则地形会旋转 90°。
#     世界原点 = -border_size（x/y 同），机器人出生点 = (-border, -border) + spawn_center_m
#     MuJoCo <hfield> 的 data 归一化到 [0,1]，z_surface = size[3](base) + size[2](zscale) * data
#       ⇒ 取 base = z_min，zscale = z_max - z_min，data = (z - z_min)/(z_max - z_min)
#     nrow/ncol = tot_rows/tot_cols（row 沿 +x，col 沿 +y；与训练侧一致）
#     size = (Lx/2, Ly/2, zscale, base)，Lx = tot_cols*h_scale, Ly = tot_rows*h_scale
#
# Usage:
#   python -m legged_gym.scripts.sim2sim_mujoco.tools.make_terrain_asset \
#       --npz legged_gym/envs/wbr/terrain/release/wbr_p8_demo.npz \
#       --base legged_gym/scripts/sim2sim_mujoco/assets/mjcf/wbr_v1.xml \
#       --name wbr_v1_demo
# ==============================================================================

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(PKG, "../../.."))
ASSETS = os.path.join(PKG, "assets")
MJCF_DIR = os.path.join(ASSETS, "mjcf")
TERRAIN_DIR = os.path.join(ASSETS, "terrain")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(npz_path, base_xml, name):
    d = np.load(npz_path, allow_pickle=False)
    hf = d["height_field_raw"].astype(np.float64)
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    hs = float(meta.get("h_scale", 0.1))
    vs = float(meta.get("v_scale", 0.002))
    border = float(meta.get("border", 25.0))
    nrow, ncol = hf.shape
    z = hf * vs
    z_min, z_max = float(z.min()), float(z.max())
    span = max(z_max - z_min, 1e-6)
    # 【公式实测校正 2026-11-06】MuJoCo hfield 表面 = geom.pos_z + size[2]×data（data∈[0,1]）；
    #   size[3]（base）只是**向下加厚的底座深度**，不是高度加法项（之前误当 +base ⇒ 整场抬高 0.7）。
    #   ⇒ pos_z = z_min、zscale = span、data = (z-z_min)/span ⇒ 表面恰等于原 npz 高度，
    #     世界坐标与训练侧/Isaac 完全一致，**不需要任何 z_offset**。base 取 0.5 托住负区。
    Z_BASE = 0.5
    Z_OFFSET = 0.0
    # npz[z] 是 (row=x, col=y)；MuJoCo PNG 按 (nrow=y, ncol=x) 行优先 ⇒ 转置后写 8-bit 灰度
    # （实测 <hfield file=...> 只认 8-bit L 模式 PNG；16-bit/其他模式会静默读成 0）
    data8 = np.round(np.clip((z - z_min) / span, 0.0, 1.0).T * 255).astype(np.uint8).copy()

    # 世界尺寸：MuJoCo ncol 沿 x、nrow 沿 y（见上），但 npz 的 row=x/col=y
    # ⇒ 物理上 x 跨度 = nrow*hs、y 跨度 = ncol*hs；hfield size[0]=x半宽、size[1]=y半宽
    Lx = nrow * hs
    Ly = ncol * hs
    # -border 是训练侧 trimesh 的 transform 原点；hfield 中心放在该区域的几何中心
    cx = -border + Lx / 2.0
    cy = -border + Ly / 2.0
    size = (Lx / 2.0, Ly / 2.0, span, Z_BASE)

    # 出生点（世界坐标）：hf 的世界起点是 -border，cell 从 hf 索引 border 开始
    #   ⇒ cell 内局部米坐标 m 的世界坐标 = -border + border*hs + m = m（与训练侧一致）。
    #   所以 spawn_center_m 就是世界坐标（demo 里 = (4.0, 3.5) = 中央十字走廊交点）。
    sc = meta.get("spawn_center_m") or [0.0, 0.0]
    spawn = (float(sc[0]), float(sc[1]))
    bpx = int(round(border / hs))
    px = min(int(round(float(sc[0]) / hs)), nrow - 1) + bpx
    py = min(int(round(float(sc[1]) / hs)), ncol - 1) + bpx
    px = min(px, nrow - 1); py = min(py, ncol - 1)
    spawn_z = float(z[px, py])

    # 生成 MJCF：复制 base_xml，把 floor plane 换成 hfield geom（保留 solref/friction）
    tree = ET.parse(base_xml)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    # MuJoCo nrow 沿 y、ncol 沿 x ⇒ 与 npz 的 (row,col) 交换
    out_png = os.path.join(TERRAIN_DIR, f"{name}.png")
    Image.fromarray(data8, mode="L").save(out_png)
    hfield = ET.SubElement(asset, "hfield", {
        "name": "terrain_hfield", "nrow": str(ncol), "ncol": str(nrow),
        "size": " ".join(f"{v:.6f}" for v in size),
        "file": os.path.abspath(out_png),   # MuJoCo 按 URDF/资源目录解析相对路径，用绝对路径最稳
    })
    world = root.find("worldbody")
    assert world is not None, "MJCF 缺 worldbody"
    floor = None
    for g in world.findall("geom"):
        if g.get("name") == "floor" or g.get("type") == "plane":
            floor = g
            break
    if floor is not None:
        # 保留地面视觉色/摩擦/solref，只改类型与 hfield 引用，并放到区域中心
        floor.set("type", "hfield")
        floor.set("hfield", "terrain_hfield")
        # plane 的 size="0 0 0.05" 对 hfield 是致命残留：实测会让高度场退化成全平
        floor.attrib.pop("size", None)
        floor.set("pos", f"{cx:.6f} {cy:.6f} {z_min:.6f}")
        # 颜色调淡（原来 0.30/0.32/0.36 太深，地形起伏看不清）
        floor.set("rgba", "0.74 0.78 0.82 1")
        fr = floor.get("friction")
        if fr is None:
            floor.set("friction", "1.0")
        sr = floor.get("solref")
        if sr is None:
            floor.set("solref", "0.01 1")
    else:
        ET.SubElement(world, "geom", {
            "name": "floor", "type": "hfield", "hfield": "terrain_hfield",
            "pos": f"{cx:.6f} {cy:.6f} 0", "friction": "1.0", "solref": "0.01 1"})
    # hfield 的底面不封口，补一块下托平面防止高速穿透后掉出世界
    ET.SubElement(world, "geom", {
        "name": "terrain_safety", "type": "plane", "size": f"{Lx:.2f} {Ly:.2f} 0.05",
        "pos": f"{cx:.6f} {cy:.6f} {z_min - Z_BASE:.4f}", "friction": "1.0",
        "contype": "1", "conaffinity": "1"})

    os.makedirs(MJCF_DIR, exist_ok=True)
    os.makedirs(TERRAIN_DIR, exist_ok=True)
    out_xml = os.path.join(MJCF_DIR, f"{name}.xml")
    out_meta = os.path.join(TERRAIN_DIR, f"{name}.json")
    ET.indent(tree, space="  ")
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
    info = {
        "source_npz": os.path.relpath(npz_path, ROOT),
        "source_sha256": sha256(npz_path),
        "base_mjcf": os.path.relpath(base_xml, ROOT),
        "mjcf": os.path.relpath(out_xml, ROOT),
        "mjcf_sha256": sha256(out_xml),
        "hfield_png": os.path.relpath(out_png, ROOT),
        "hfield_png_sha256": sha256(out_png),
        "hfield_data_shape": [ncol, nrow],  # MuJoCo (nrow_y, ncol_x) = (npz_col, npz_row)
        "quant_mm": round(span * 1000 / 255, 3),
        "h_scale": hs, "v_scale": vs, "border": border,
        "size": list(size), "center_xy": [cx, cy],
        "z_min": z_min, "z_max": z_max,
        "z_offset": Z_OFFSET, "z_base": Z_BASE, "z_pos": z_min,
        "spawn_xy": list(spawn), "spawn_z": spawn_z + Z_OFFSET,
        "spawn_center_m": sc,
        "note": ("高度图以 8-bit L 模式 PNG 内嵌在 <hfield file=...>（MuJoCo 3.17 只认该格式）；"
                 "z_surface = z_base + zscale*data = z + z_offset；机器人初始 z 需加 z_offset"),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "legged_gym/envs/wbr/terrain/release/wbr_p8_demo.npz"))
    ap.add_argument("--base", default=os.path.join(MJCF_DIR, "wbr_v1.xml"))
    ap.add_argument("--name", default="wbr_v1_demo")
    a = ap.parse_args()
    info = build(a.npz, a.base, a.name)
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
