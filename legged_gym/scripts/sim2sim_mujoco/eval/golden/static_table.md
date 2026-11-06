# 四轮静态对账（MJCF ↔ Isaac）

- MJCF：`legged_gym/scripts/sim2sim_mujoco/assets/mjcf/wbr_v2.xml`
- Isaac：`legged_gym/scripts/sim2sim_mujoco/eval/golden/isaac_static.json`
- 容差：1e-06（惯量主值 1e-9）

| 项 | MJCF | Isaac | 差 | 判定 |
|---|---|---|---|---|
| 总质量 (kg) | 7.223110 | 7.223110 | 2.19e-09 | PASS |
| base_link.mass | 2.541800 | 2.541800 | 2.21e-08 | PASS |
| base_link.com_x | 0.000044 | 0.000044 | 1.01e-12 | PASS |
| base_link.com_y | 0.000116 | 0.000116 | 1.55e-13 | PASS |
| base_link.com_z | 0.010416 | 0.010416 | 1.50e-10 | PASS |
| base_link.I_min | 0.010577600 | 0.010577611 | 1.14e-08 | PASS |
| base_link.I_mid | 0.013713000 | 0.013712987 | 1.29e-08 | PASS |
| base_link.I_max | 0.015791400 | 0.015791404 | 3.60e-09 | PASS |
| left_thigh_link.mass | 0.808150 | 0.808150 | 6.58e-09 | PASS |
| left_thigh_link.com_x | 0.156590 | 0.156590 | 2.05e-10 | PASS |
| left_thigh_link.com_y | 0.000320 | 0.000320 | 7.70e-12 | PASS |
| left_thigh_link.com_z | 0.020250 | 0.020250 | 5.96e-11 | PASS |
| left_thigh_link.I_min | 0.000587497 | 0.000587497 | 3.30e-10 | PASS |
| left_thigh_link.I_mid | 0.004768000 | 0.004768002 | 1.98e-09 | PASS |
| left_thigh_link.I_max | 0.005153610 | 0.005153612 | 2.28e-09 | PASS |
| left_shank_link.mass | 0.931380 | 0.931380 | 2.61e-08 | PASS |
| left_shank_link.com_x | 0.121560 | 0.121560 | 1.17e-10 | PASS |
| left_shank_link.com_y | 0.004872 | 0.004872 | 1.93e-10 | PASS |
| left_shank_link.com_z | 0.019960 | 0.019960 | 7.52e-10 | PASS |
| left_shank_link.I_min | 0.000792105 | 0.000792104 | 6.32e-10 | PASS |
| left_shank_link.I_mid | 0.007251880 | 0.007251886 | 6.04e-09 | PASS |
| left_shank_link.I_max | 0.007802530 | 0.007802531 | 7.75e-10 | PASS |
| left_knee_caster.mass | 0.050000 | 0.050000 | 7.45e-10 | PASS |
| left_knee_caster.com_x | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| left_knee_caster.com_y | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| left_knee_caster.com_z | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| left_knee_caster.I_min | 0.000016820 | 0.000016820 | 3.83e-13 | PASS |
| left_knee_caster.I_mid | 0.000016820 | 0.000016820 | 3.83e-13 | PASS |
| left_knee_caster.I_max | 0.000016820 | 0.000016820 | 3.83e-13 | PASS |
| left_wheel_link.mass | 0.553360 | 0.553360 | 1.46e-08 | PASS |
| left_wheel_link.com_x | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| left_wheel_link.com_y | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| left_wheel_link.com_z | 0.025381 | 0.025381 | 7.13e-10 | PASS |
| left_wheel_link.I_min | 0.001090200 | 0.001090200 | 3.58e-11 | PASS |
| left_wheel_link.I_mid | 0.001090200 | 0.001090200 | 3.58e-11 | PASS |
| left_wheel_link.I_max | 0.001974500 | 0.001974500 | 1.72e-11 | PASS |
| right_thigh_link.mass | 0.808150 | 0.808150 | 6.58e-09 | PASS |
| right_thigh_link.com_x | 0.156590 | 0.156590 | 2.05e-10 | PASS |
| right_thigh_link.com_y | 0.000190 | 0.000190 | 3.32e-12 | PASS |
| right_thigh_link.com_z | 0.020250 | 0.020250 | 5.96e-11 | PASS |
| right_thigh_link.I_min | 0.000587476 | 0.000587476 | 1.56e-10 | PASS |
| right_thigh_link.I_mid | 0.004768080 | 0.004768083 | 3.19e-09 | PASS |
| right_thigh_link.I_max | 0.005153710 | 0.005153710 | 6.78e-11 | PASS |
| right_shank_link.mass | 0.926910 | 0.926910 | 4.28e-08 | PASS |
| right_shank_link.com_x | 0.121140 | 0.121140 | 3.20e-09 | PASS |
| right_shank_link.com_y | 0.005186 | 0.005186 | 2.07e-10 | PASS |
| right_shank_link.com_z | 0.020146 | 0.020146 | 7.32e-10 | PASS |
| right_shank_link.I_min | 0.000785120 | 0.000785120 | 1.19e-11 | PASS |
| right_shank_link.I_mid | 0.007210870 | 0.007210867 | 2.95e-09 | PASS |
| right_shank_link.I_max | 0.007760930 | 0.007760933 | 2.98e-09 | PASS |
| right_knee_caster.mass | 0.050000 | 0.050000 | 7.45e-10 | PASS |
| right_knee_caster.com_x | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| right_knee_caster.com_y | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| right_knee_caster.com_z | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| right_knee_caster.I_min | 0.000016820 | 0.000016820 | 3.83e-13 | PASS |
| right_knee_caster.I_mid | 0.000016820 | 0.000016820 | 3.83e-13 | PASS |
| right_knee_caster.I_max | 0.000016820 | 0.000016820 | 3.83e-13 | PASS |
| right_wheel_link.mass | 0.553360 | 0.553360 | 1.46e-08 | PASS |
| right_wheel_link.com_x | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| right_wheel_link.com_y | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| right_wheel_link.com_z | 0.025381 | 0.025381 | 7.13e-10 | PASS |
| right_wheel_link.I_min | 0.001090200 | 0.001090200 | 3.58e-11 | PASS |
| right_wheel_link.I_mid | 0.001090200 | 0.001090200 | 3.58e-11 | PASS |
| right_wheel_link.I_max | 0.001974500 | 0.001974500 | 1.72e-11 | PASS |
| 关节集合 | 8 个 | 8 个 | 缺=[] 多=[] | PASS |
| left_hip_joint.damping | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| left_hip_joint.actuatorfrc | 23.700 | 23.700 | 7.63e-07 | PASS |
| left_hip_joint.lower | -1.570800 | -1.570800 | 5.32e-08 | PASS |
| left_hip_joint.upper | 1.570800 | 1.570800 | 5.32e-08 | PASS |
| left_knee_joint.damping | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| left_knee_joint.actuatorfrc | 23.700 | 23.700 | 7.63e-07 | PASS |
| left_knee_joint.lower | -2.705300 | -2.705300 | 9.27e-08 | PASS |
| left_knee_joint.upper | -0.100000 | -0.100000 | 1.49e-09 | PASS |
| left_knee_caster_joint.damping | 0.010000 | 0.010000 | 2.24e-10 | PASS |
| left_knee_caster_joint.actuatorfrc | 1.000 | 1.000 | 0.00e+00 | PASS |
| left_wheel_joint.damping | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| left_wheel_joint.actuatorfrc | 23.700 | 23.700 | 7.63e-07 | PASS |
| right_hip_joint.damping | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| right_hip_joint.actuatorfrc | 23.700 | 23.700 | 7.63e-07 | PASS |
| right_hip_joint.lower | -1.570800 | -1.570800 | 5.32e-08 | PASS |
| right_hip_joint.upper | 1.570800 | 1.570800 | 5.32e-08 | PASS |
| right_knee_joint.damping | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| right_knee_joint.actuatorfrc | 23.700 | 23.700 | 7.63e-07 | PASS |
| right_knee_joint.lower | 0.100000 | 0.100000 | 1.49e-09 | PASS |
| right_knee_joint.upper | 2.705300 | 2.705300 | 9.27e-08 | PASS |
| right_knee_caster_joint.damping | 0.010000 | 0.010000 | 2.24e-10 | PASS |
| right_knee_caster_joint.actuatorfrc | 1.000 | 1.000 | 0.00e+00 | PASS |
| right_wheel_joint.damping | 0.000000 | 0.000000 | 0.00e+00 | PASS |
| right_wheel_joint.actuatorfrc | 23.700 | 23.700 | 7.63e-07 | PASS |

**总判定：PASS**（0 项未过）
