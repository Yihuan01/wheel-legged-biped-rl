# resources/robots/wbr —— WBR 机器人资产

| 路径 | 用途 |
|---|---|
| `urdf/wbr_v1.urdf` | 两轮构型 URDF（`wbr_flat` / `wbr_terrain` 等使用） |
| `urdf/wbr_v2.urdf` | 四轮构型 URDF（主动轮 + 球形从动轮），`flat4` / `switch` / `uni` 使用 |
| `meshes_v1/` | 两份 URDF 共用的网格（base、大腿、小腿、轮） |

几何校验脚本已移到 `legged_gym/scripts/check_geometry_4w.py`（与 `check_geometry_figs.py` 配对）。

配置里的 `asset.file` 写的是 `{LEGGED_GYM_ROOT_DIR}/resources/robots/wbr/urdf/wbr_v1.urdf`（或 `wbr_v2.urdf`），
新增 URDF 时记得同步改配置与 `.gitignore`（历史实验版不要入库）。
