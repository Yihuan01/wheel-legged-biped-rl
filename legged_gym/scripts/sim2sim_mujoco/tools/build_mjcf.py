# ==============================================================================
# Date      : 2026-11-05
# Author    : SYH
# File      : build_mjcf.py
# Purpose   : 由 wbr_v2.urdf 生成四轮部署用 MJCF：补浮动基座与地面、钉住物理选项与接触参数，
# 从动轮摩擦按训练名义值写入；生成物带 sha256，静态对账以它为基准。
# ==============================================================================

import argparse
import hashlib
import math
import os
import re
import xml.etree.ElementTree as ET

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(PKG, "../../.."))   # sim2sim_mujoco → scripts → legged_gym → 仓库根

# 四轮名义值（源：wbr_flat4_config.py / wbr_flat4.py）
CASTER_MU = 0.02        # caster_friction_range 下沿＝无随机
GROUND_MU = 1.0         # terrain.static_friction / dynamic_friction
# 接触求解时间常数：默认 0.02 在动态闭环里太软 ⇒ vx 差 −13.5%/+17%（walk03/06）；
# 收紧到 0.01（MuJoCo 钳到 2×timestep，即实际生效下限）后 walk03 +4.8%、walk06 +9.7%，S1 判据全绿，
# 零动作沉降 0.9 mm（静态几乎不敏感），已入冻结资产。floor/轮/从动轮同设，规避 geom 组合规则。
SOLREF_TC = 0.01
DT = 0.005              # sim.dt
GRAVITY = (0.0, 0.0, -9.81)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rpy_quat(rpy):
    """URDF rpy（固定轴 XYZ）→ MuJoCo 四元数 (w,x,y,z)。"""
    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return [cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def parse_visuals(urdf):
    """抽 <visual>：link → (mesh 文件名, xyz, rpy, rgba)。STL 目录由 mesh 路径推。"""
    root = ET.parse(urdf).getroot()
    out = {}
    for link in root.findall("link"):
        vis = link.find("visual")
        if vis is None:
            continue
        origin = vis.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split())] if origin is not None else [0, 0, 0]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0").split())] if origin is not None else [0, 0, 0]
        geo = vis.find("geometry")
        mesh = geo.find("mesh") if geo is not None else None
        color = vis.find("material/color")
        rgba = [float(v) for v in color.get("rgba").split()] if color is not None else [0.75, 0.75, 0.75, 1.0]
        out[link.get("name")] = {
            "mesh": os.path.basename(mesh.get("filename")) if mesh is not None else None,
            "meshdir": os.path.dirname(mesh.get("filename")) if mesh is not None else None,
            "xyz": xyz, "rpy": rpy, "rgba": rgba,
        }
    return out


def build(urdf, out, mesh=True):
    spec = mujoco.MjSpec.from_file(urdf)
    # URDF 的 base_link 默认被焊到 world；四轮必须自由浮动。
    spec.body("base_link").add_freejoint()
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.friction = [GROUND_MU, 0.005, 0.0001]
    floor.solref = [SOLREF_TC, 1.0]
    floor.rgba = [0.30, 0.32, 0.36, 1.0]

    spec.option.timestep = DT
    spec.option.gravity = list(GRAVITY)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.impratio = 1.0
    spec.option.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    # 离屏渲染尺寸（只影响渲染，不影响物理）：方便出图/录视频
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    # 视觉网格的默认灯光偏暗，提一档（只影响显示/渲染）
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.visual.headlight.specular = [0.25, 0.25, 0.25]

    # 从动轮球：低侧向摩擦（名义 μ=0.02）；地面/驱动轮保持 1.0
    # 轮/从动轮/地面同设 solref（S1b：动态接触刚度对齐 PhysX 行为，见文件头注释）
    for g in spec.geoms:
        body = g.parent.name if g.parent is not None else ""
        if "caster" in body:
            g.friction = [CASTER_MU, 0.005, 0.0001]
            g.solref = [SOLREF_TC, 1.0]
            g.rgba = [0.85, 0.55, 0.20, 1.0]
        elif "wheel_link" in body:
            g.solref = [SOLREF_TC, 1.0]

    # 碰撞体挪到 group 3（默认不显示），视觉网格放 group 2（默认显示）。
    for g in spec.geoms:
        if g.name == "floor":
            continue
        g.group = 3

    if mesh:
        vis = parse_visuals(urdf)
        mesh_dir = None
        for link, v in vis.items():
            if v["mesh"]:
                mesh_dir = os.path.abspath(os.path.join(os.path.dirname(urdf), v["meshdir"]))
                break
        if mesh_dir is None:
            mesh_dir = os.path.abspath(os.path.join(os.path.dirname(urdf), "../meshes_v1"))
        spec.meshdir = mesh_dir
        for link, v in vis.items():
            try:
                body = spec.body(link)
            except Exception:
                continue
            if v["mesh"]:
                me = spec.add_mesh()
                me.name = f"{link}_visual"
                me.file = v["mesh"]
                g = body.add_geom()
                g.name = f"{link}_vis"
                g.type = mujoco.mjtGeom.mjGEOM_MESH
                g.meshname = me.name
                g.pos = v["xyz"]
                g.quat = _rpy_quat(v["rpy"])
                g.rgba = v["rgba"]
                g.group = 2
            else:
                # 从动轮没有网格：给一个同尺寸的可视球（碰撞球仍在 group 3）
                g = body.add_geom()
                g.name = f"{link}_vis"
                g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                g.size = [0.029, 0.0, 0.0]
                g.rgba = [0.85, 0.55, 0.20, 1.0]
                g.group = 2

    # ⚠ 必须先 compile 再 to_xml：MjSpec 只有在编译后才会把 mesh 资产与 mesh geom 序列化出来
    # （只 to_xml 的话整段 <asset> 与视觉 geom 会被静默丢掉）。
    m = spec.compile()
    xml = spec.to_xml()
    # ⚠ 不用 MjSpec 直接设 contype=0＋conaffinity=0：两者同时为 0 时 compile() 会把该 geom 整条丢掉；
    # 写成 XML 属性则正常保留（视觉几何不参与碰撞）。
    xml = re.sub(r'(<geom name="[^"]*_vis")', r'\1 contype="0" conaffinity="0"', xml)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    return m


def relative_meshdir(out):
    """把 mesh 路径改成相对 meshdir，保证同一份 MJCF 在两个工作区都能编译。"""
    xml = open(out, encoding="utf-8").read()
    paths = re.findall(r'<mesh file="([^"]+)"', xml)
    if not paths:
        return xml
    mesh_dir = os.path.dirname(os.path.abspath(paths[0]))
    rel = os.path.relpath(mesh_dir, os.path.dirname(os.path.abspath(out)))

    idx = [0]

    def repl(_match):
        name = os.path.basename(paths[idx[0]])
        idx[0] += 1
        return f'<mesh file="{name}"'

    xml = re.sub(r'<mesh file="[^"]+"', repl, xml)
    if re.search(r'meshdir="[^"]*"', xml):
        xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{rel}"', xml, count=1)
    elif "<compiler" in xml:
        xml = re.sub(r"<compiler", f'<compiler meshdir="{rel}"', xml, count=1)
    else:
        xml = re.sub(r"(<mujoco[^>]*>)", rf'\1\n  <compiler meshdir="{rel}"/>', xml, count=1)
    return xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=os.path.join(ROOT, "resources/robots/wbr/urdf/wbr_v2.urdf"))
    ap.add_argument("--out", default=os.path.join(PKG, "assets/mjcf/wbr_v2.xml"))
    ap.add_argument("--no-mesh", action="store_true", help="只生成碰撞体（老版本，窗口里是方块）")
    a = ap.parse_args()

    build(a.urdf, a.out, mesh=not a.no_mesh)
    xml = relative_meshdir(a.out)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(xml)

    m2 = mujoco.MjModel.from_xml_path(a.out)
    print(f"[ok] {a.out}")
    print(f"     nq={m2.nq} nv={m2.nv} njnt={m2.njnt} ngeom={m2.ngeom} "
          f"mass={float(m2.body_mass.sum()):.5f} kg dt={m2.opt.timestep}")
    print(f"     urdf sha256={sha256(a.urdf)}")
    print(f"     mjcf sha256={sha256(a.out)}")


if __name__ == "__main__":
    main()
