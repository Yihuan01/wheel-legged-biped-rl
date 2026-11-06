/* ============================================================================
   WBR Console — 前端逻辑（P2：接后端）
   数据来自 wbr_bench.py 的 /api/*；若后端不可达（直接双击 app.html），
   自动退回 mock.js 的离线预览，并在顶栏标注「离线预览」。
   ========================================================================== */
(function () {
  'use strict';

  var app = document.getElementById('app');
  var OFFLINE = false;
  var MOCK = window.MOCK || null;

  /* ------------------------------------------------------------------ 工具 */
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function el(tag, attrs, children) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === 'class') e.className = attrs[k];
      else if (k === 'html') e.innerHTML = attrs[k];
      else if (k === 'text') e.textContent = attrs[k];
      else if (k.slice(0, 2) === 'on') e.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
    }
    (children || []).forEach(function (c) {
      if (c === null || c === undefined || c === false) return;
      e.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
    });
    return e;
  }
  function fmt(v, d) { var t = (+v).toFixed(d == null ? 2 : d); return /^-0(\.0+)?$/.test(t) ? t.slice(1) : t; }
  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
  function nowStr() {
    var d = new Date();
    function p(x) { return String(x).padStart(2, '0'); }
    return p(d.getMonth() + 1) + '-' + p(d.getDate()) + ' ' + p(d.getHours()) + ':' + p(d.getMinutes());
  }
  // 条目图标按钮（SVG，描边继承 currentColor）：笔=覆写本条，垃圾桶=删除本条
  function iconSvg(pathHtml) {
    var w = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    w.setAttribute('viewBox', '0 0 24 24');
    w.setAttribute('fill', 'none');
    w.setAttribute('stroke', 'currentColor');
    w.setAttribute('stroke-width', '1.9');
    w.setAttribute('stroke-linecap', 'round');
    w.setAttribute('stroke-linejoin', 'round');
    w.innerHTML = pathHtml;
    return w;
  }
  function iconBtn(kind, title, onclick) {
    var pen = '<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4z"/>';
    var trash = '<path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>'
      + '<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/>';
    var b = el('button', {
      class: 'iconbtn' + (kind === 'del' ? ' del' : ''), title: title, type: 'button',
      onclick: function (e) { if (e) e.stopPropagation(); onclick(); },
      ondblclick: function (e) { if (e) e.stopPropagation(); }
    }, [iconSvg(kind === 'del' ? trash : pen)]);
    return b;
  }

  var store = {
    get: function (k, d) { try { var v = localStorage.getItem('wbr.' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set: function (k, v) { try { localStorage.setItem('wbr.' + k, JSON.stringify(v)); } catch (e) { } }
  };

  function toast(msg, level) {
    var box = document.getElementById('toast');
    var t = el('div', { class: 'pill', text: msg });
    t.style.cssText = 'background:var(--card);box-shadow:0 8px 24px rgba(0,0,0,.16);color:var(--foreground);border-color:' +
      (level === 'warn' ? 'var(--warning)' : level === 'bad' ? 'var(--destructive)' : level === 'ok' ? 'var(--success)' : 'var(--primary)');
    box.appendChild(t);
    setTimeout(function () { t.style.opacity = '0'; t.style.transition = 'opacity .3s'; }, 1800);
    setTimeout(function () { t.remove(); }, 2300);
  }

  /* ------------------------------------------------------------------ API */
  function api(path, opt) {
    opt = opt || {};
    return fetch(path, {
      method: opt.method || 'GET',
      headers: { 'Content-Type': 'application/json' },
      body: opt.body ? JSON.stringify(opt.body) : undefined
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok) { var e = new Error(j.text || j.error || ('HTTP ' + r.status)); e.payload = j; throw e; }
        return j;
      });
    });
  }
  var GET = function (p) { return api(p); };
  var POST = function (p, b) { return api(p, { method: 'POST', body: b }); };
  var DEL = function (p, b) { return api(p, { method: 'DELETE', body: b }); };

  /* --------------------------------------------------------------- 全局状态 */
  var S = {
    mode: 'uni',
    engine: store.get('engine', 'isaac'),   // 【Sim2Sim P2】启动引擎选择；缺省 isaac（向后兼容）
    tab: (function () { try { var t = new URLSearchParams(location.search).get('tab'); if (t) return t; } catch (e) { } return store.get('tab', 'run'); })(),
    navpos: store.get('navpos', 'left'),
    theme: store.get('theme', 'system'),
    slots: null,          // {fourw:{path,obs,act,model,ckpt}, ...}
    single: { task: 'wbr_flat', run: '', checkpoint: null, from_run_config: '', path: '', robots: 3 },
    params: {},  // 衔接参数（键值见 PARAM_DEFS/PARAM_DEFAULTS，缺省时用默认值）
    scenarios: [],
    saved: [],
    checkpoints: [],
    runs: [],
    settings: {},
    state: null,          // /api/state
    sim: { status: 'idle' },
    pickerSlot: null,
    saveCtx: null,        // 保存弹窗上下文：{kind:'scenario'|'saved', name?, forName?}
    delCtx: null,         // 删除确认上下文：{kind:'scenario'|'saved', name, label}
    plotLog: '',
    telLogs: []
  };

  var TABS = [{ key: 'run', cn: '仿真运行' }, { key: 'model', cn: '模型配置' }, { key: 'mujoco', cn: 'MuJoCo 配置' }, { key: 'plot', cn: '数据绘图' }, { key: 'use', cn: '使用说明' }];
  // 【Sim2Sim P2】MuJoCo 场景预设（设计稿 §6）；h 量程按契约 [0.175, 0.245]
  var MJ_PRESETS = {
    stand: { vx: 0, yaw: 0, h: 0.19, label: '站定 0.19' }, walk03: { vx: 0.3, yaw: 0, h: 0.21, label: '走 0.3' },
    walk06: { vx: 0.6, yaw: 0, h: 0.21, label: '走 0.6' }, h0175: { vx: 0, yaw: 0, h: 0.175, label: '高度 0.175' },
    h023: { vx: 0, yaw: 0, h: 0.23, label: '高度 0.23' }, h0245: { vx: 0, yaw: 0, h: 0.245, label: '高度 0.245' },
    yaw05: { vx: 0.3, yaw: 0.5, h: 0.21, label: '转向 0.5' }
  };
  // 【任务③ 两轮】两轮场景预设（对齐 parity_two 场景：stand/walk03/walk06/yaw05/h020/h035）
  var MJ_PRESETS_TWO = {
    stand: { vx: 0, yaw: 0, h: 0.25, label: '站定 0.25' }, walk03: { vx: 0.3, yaw: 0, h: 0.25, label: '走 0.3' },
    walk06: { vx: 0.6, yaw: 0, h: 0.25, label: '走 0.6' }, yaw05: { vx: 0.3, yaw: 0.5, h: 0.25, label: '转向 0.5' },
    h020: { vx: 0, yaw: 0, h: 0.20, label: '高度 0.20' }, h035: { vx: 0, yaw: 0, h: 0.35, label: '高度 0.35' }
  };
  // 【S3 槽位化】每个槽＝一套契约＋资产＋权重；uni/uni4 为五槽链（wbr_uni 同序）
  var MJ_SLOTS = {
    four: { cn: '四轮 164', contract: 'legged_gym/scripts/sim2sim_mujoco/assets/deploy_contract.json',
            mjcf: 'legged_gym/scripts/sim2sim_mujoco/assets/mjcf/wbr_v2.xml', obs: 164, defH: 0.19,
            presets: 'four', roll: false, jump: false, release: 'P7F3e_yaw1_dr15' },
    two: { cn: '两轮 161', contract: 'legged_gym/scripts/sim2sim_mujoco/assets/deploy_contract_two.json',
           mjcf: 'legged_gym/scripts/sim2sim_mujoco/assets/mjcf/wbr_v1.xml', obs: 161, defH: 0.25,
           presets: 'two', roll: true, jump: false, release: 'P8W7_push01_fric05' },
    jump: { cn: '两轮跳跃 161', contract: 'legged_gym/scripts/sim2sim_mujoco/assets/deploy_contract_two.json',
            mjcf: 'legged_gym/scripts/sim2sim_mujoco/assets/mjcf/wbr_v1.xml', obs: 161, defH: 0.25,
            presets: 'two', roll: true, jump: true, release: 'P8J3d_wheelfollow_lite' },
    lie: { cn: '趴下 173', contract: 'legged_gym/scripts/sim2sim_mujoco/assets/deploy_contract_lie.json',
           mjcf: 'legged_gym/scripts/sim2sim_mujoco/assets/mjcf/wbr_v2.xml', obs: 173, defH: 0.19,
           presets: 'four', roll: false, jump: false, release: 'P7L_liedown' },
    uni: { cn: '五槽链（两轮起步）', chain: true, start: '2w', obs: 0, defH: 0.25, presets: 'two' },
    uni4: { cn: '五槽链（四轮起步）', chain: true, start: '4w', obs: 0, defH: 0.19, presets: 'four' }
  };
  function mjSlotKey() { return MJ_SLOTS[S.mj.slot] ? S.mj.slot : 'four'; }
  function mjSlot() { return MJ_SLOTS[mjSlotKey()]; }
  function mjPresets() { return mjSlot().presets === 'two' ? MJ_PRESETS_TWO : MJ_PRESETS; }
  function mjChain() { return !!mjSlot().chain; }
  var MJ_CFG_KEY = 'wbr.mj.cfg';
  function mjLoadCfg() {
    var cfg = {};
    try { cfg = JSON.parse(localStorage.getItem(MJ_CFG_KEY) || '{}') || {}; } catch (e) { cfg = {}; }
    // 旧配置没有 terrain 字段 ⇒ 默认按「复杂地形」（作者 2026-11-06 口径：五槽应接复杂地形）
    if (cfg.terrain === undefined) cfg.terrain = true;
    return cfg;
  }
  S.mj = Object.assign({ slot: 'four', terrain: true, release: '', model: '', customPt: '', mjcf: '', preset: 'walk03', vx: 0, yaw: 0, h: 0.21, roll: 0, dur: 0, record: true, window: true, speed: 1 }, mjLoadCfg());
  var CHANNELS = [
    { key: 'lin_vel_x', label: 'Vx', unit: 'm/s', min: -1, max: 1, step: .05, def: 0, d: 2, short: 'vx' },
    { key: 'ang_vel_yaw', label: 'Yaw', unit: 'rad/s', min: -1, max: 1, step: .05, def: 0, d: 2, short: 'yaw' },
    { key: 'base_height', label: 'Height', unit: 'm', min: .175, max: .35, step: .005, def: .19, d: 3, short: 'h' },
    { key: 'ang_roll', label: 'Roll', unit: 'rad', min: -.2, max: .2, step: .02, def: 0, d: 2, short: 'roll' },
    { key: 'jump_height', label: 'Jump', unit: 'm', min: 0, max: .15, step: .01, def: 0, d: 2, short: 'jump', switchOff: true }
  ];
  var SLOT_ORDER = ['fourw', 'standup', 'w3', 'lie', 'jump'];
  var SLOT_STAGE = { fourw: '① 四轮行驶', standup: '② 起立', w3: '③ 两轮站立/行走', lie: '④ 趴下', jump: '⑤ 跳跃' };
  var SLOT_OBS = { fourw: 164, standup: 173, w3: 161, lie: 173, jump: 161 };
  // 起立槽可为「相位脚本」（P7T，无权重）：场景里写作 {"type":"script"}，预检与运行端都认（4/4 照旧）。
  var SCRIPT_SLOT_LABEL = '相位脚本（P7T，无需权重）';
  function isScriptSlot(v) { return !!v && typeof v === 'object' && String(v.type || '').toLowerCase() === 'script'; }
  function scriptSlotObj() { return { path: null, script: true, model: SCRIPT_SLOT_LABEL, ckpt: '—', obs: null, act: null }; }
  // wbr_terrain 与 wbr_flat 观测同为 161（无高程观测）⇒ INV 只保留一一对应的维度，
  // 161 的反查走 taskForObs（带路径判断）。
  var TASK_OBS = { wbr_flat: 161, wbr_jump: 161, wbr_flat4: 164, wbr_switch: 173, wbr_terrain: 161,
                   wbr_demo: 161, wbr_combo: 161, wbr_elements: 161 };
  var TASK_OBS_INV = { 161: 'wbr_flat', 164: 'wbr_flat4', 173: 'wbr_switch' };
  // 四槽式模式（switch / uni）共用同一套四槽 checkpoint 与主循环；single 为单策略。
  function isSlotMode(m) {
    // 【MuJoCo】链模式的通道可用性由后端 cmdRanges 表达（四轮 jump 量程退化 ⇒ 自动置灰），
    //   不能沿用 Isaac 的「仅 TWOW 可跳」门控——那要求 st.modeKey，而 MuJoCo 桥不返回它，
    //   结果 jump 滑条永远置灰（作者 2026-11-06 反馈）。
    //   用桥上报的**实际引擎**（S.engine 只是用户选择，从配置页一键启动时仍是 isaac）
    if ((S.state && S.state.engine === 'mujoco') || S.engine === 'mujoco') return false;
    m = m || S.mode; return m === 'uni';
  }
  var MODE_LABEL = { uni: '双模式', single: '单策略' };
  // 衔接参数：standup_mode 选起立方式（p7s 模型 / trad 传统脚本链）；两套起立参数互斥（非当前方式置灰）。
  //   leg_p_gains 是 P7S 起立伸腿/交接期的腿 P 增益倍率（wbr_bench_run 包装 playback_set_mode 实现）。
  var PARAM_DEFS = [
    { key: 'standup_mode', label: '起立方式', type: 'select', opts: [['p7s', '模型（P7S）'], ['trad', '传统（P7T）']] },
    { key: 't_align', label: '对齐', unit: 's', min: 0, max: 3, step: 0.05, d: 2 },
    { key: 't_rise', label: '起立', unit: 's', min: 0, max: 3, step: 0.05, d: 2 },
    { key: 't_handover', label: '交接', unit: 's', min: 0, max: 3, step: 0.05, d: 2 },
    { key: 'leg_p_gains', label: '腿P倍率', unit: '×', min: 1, max: 4, step: 0.1, d: 1 },
    { key: 't_lower', label: '降高', unit: 's', min: 0, max: 3, step: 0.05, d: 2 },
    { key: 't_tilt', label: '倾转', unit: 's', min: 0, max: 3, step: 0.05, d: 2 },
    { key: 't_absorb', label: '缓冲', unit: 's', min: 0, max: 3, step: 0.05, d: 2 },
    { key: 't_trad_hold', label: '静稳', unit: 's', min: 0, max: 3, step: 0.05, d: 2 },
    { key: 't_trad_crouch', label: '折蹲', unit: 's', min: 0, max: 3, step: 0.05, d: 2 },
    { key: 't_trad_crouch_hold', label: '蹲稳', unit: 's', min: 0, max: 3, step: 0.05, d: 2 },
    { key: 't_trad_extend', label: '伸腿', unit: 's', min: 0, max: 3, step: 0.05, d: 2 },
    { key: 't_trad_lift_max', label: '抬升上限', unit: 's', min: 0.2, max: 5, step: 0.1, d: 1 },
    { key: 'trad_torque', label: '轮力矩C', unit: 'N·m', min: 4, max: 12, step: 0.5, d: 1 },
    { key: 'trad_boost', label: '腿P(伸腿)', unit: '×', min: 1, max: 3, step: 0.1, d: 1 },
    { key: 'trad_hcmd', label: '交体高', unit: 'm', min: 0.18, max: 0.28, step: 0.01, d: 2 },
    { key: 'trad_hcmd_hold', label: '体高保持', unit: 's', min: 0, max: 10, step: 0.5, d: 1 }
  ];
  // 参数按「生效位置」分组排版；when 指定该组仅在某起立方式下生效（否则置灰）。
  var PARAM_GROUPS = [
    { title: '② 起立 · 模型法 (P7S)', hint: '起立方式=模型 时生效', when: 'p7s',
      keys: ['t_rise', 't_handover', 'leg_p_gains'] },
    { title: '② 起立 · 传统法 (P7T)', hint: '起立方式=传统 时生效', when: 'trad',
      keys: ['t_trad_hold', 't_trad_crouch', 't_trad_crouch_hold', 't_trad_extend',
             't_trad_lift_max', 'trad_torque', 'trad_boost', 'trad_hcmd', 'trad_hcmd_hold'] },
    { title: '④ 趴下', hint: '两轮→四轮 过渡时序', when: null,
      keys: ['t_lower', 't_tilt', 't_absorb'] },
    { title: '通用 · 对齐', hint: '起立/趴下起点共用', when: null,
      keys: ['t_align'] }
  ];
  // 与 wbr4_switch_config.py 的 switch 默认值一致（首次启动前的初值）；trad 默认＝2026-10-25 harness 验证值
  var PARAM_DEFAULTS = {
    t_align: 0.30, t_rise: 0.60, t_handover: 0.50, t_lower: 0.50, t_tilt: 0.30, t_absorb: 0.40,
    standup_mode: 'p7s', leg_p_gains: 1.0,
    t_trad_hold: 0.60, t_trad_crouch: 0.50, t_trad_crouch_hold: 0.40, t_trad_extend: 0.50,
    t_trad_lift_max: 1.2, trad_torque: 9.0, trad_boost: 2.0, trad_hcmd: 0.22, trad_hcmd_hold: 3.0
  };

  /* ------------------------------------------------------------- 导航渲染 */
  // 顶栏右侧控制块（仿真状态 / 导航位置 / 配色）：可整体在「顶部 / 左侧 / 右侧」之间搬迁。
  // 放在 #topCtrls 容器里；左/右侧导航模式下容器被移进侧栏（.rail）并竖排。
  function renderTopCtrls() {
    var host = document.getElementById('topCtrls');
    host.innerHTML = '';
    host.appendChild(el('div', { class: 'state', id: 'simStateWrap' }, [
      el('i', { class: 'dot', id: 'simDot' }),
      el('span', { id: 'simStateText', text: '仿真未启动' })
    ]));
    host.appendChild(el('div', { class: 'seg', id: 'navposSeg', title: '导航位置' }, [
      el('button', { 'data-navpos': 'top', 'aria-pressed': String(S.navpos === 'top'), text: '顶部导航' }),
      el('button', { 'data-navpos': 'left', 'aria-pressed': String(S.navpos === 'left'), text: '左侧导航' }),
      el('button', { 'data-navpos': 'right', 'aria-pressed': String(S.navpos === 'right'), text: '右侧导航' })
    ]));
    host.appendChild(el('div', { class: 'seg theme-seg', id: 'themeSeg', title: '配色方案' }, [
      el('button', { 'data-theme': 'light', 'aria-pressed': String(S.theme === 'light'), title: '浅色', text: '🌞 浅色' }),
      el('button', { 'data-theme': 'dark', 'aria-pressed': String(S.theme === 'dark'), title: '深色', text: '🌙 深色' }),
      el('button', { 'data-theme': 'system', 'aria-pressed': String(S.theme === 'system'), title: '跟随系统', text: '🖥 跟随' })
    ]));
    // 重新绑定（块是新建的）
    $$('#navposSeg [data-navpos]').forEach(function (b) { b.addEventListener('click', function () { setNavPos(b.getAttribute('data-navpos')); }); });
    $$('#themeSeg [data-theme]').forEach(function (b) { b.addEventListener('click', function () { applyTheme(b.getAttribute('data-theme')); }); });
  }

  // 保证 #topCtrls 按 S.navpos 落在正确容器：顶部在顶栏（且顶栏可见），左/右在侧栏（顶栏隐藏）
  function ensureNavTopCtrls() {
    var ctrls = document.getElementById('topCtrls');
    var rail = document.getElementById('navRail');
    var topbar = document.querySelector('.topbar');
    if (!ctrls || !rail || !topbar) return;
    if (S.navpos === 'top') {
      if (ctrls.parentNode !== topbar) topbar.appendChild(ctrls);
      ctrls.classList.remove('in-rail');
    } else {
      if (ctrls.parentNode !== rail) rail.appendChild(ctrls);
      ctrls.classList.add('in-rail');
    }
  }

  function renderNav() {
    var navTop = document.getElementById('navTop');
    var rail = document.getElementById('navRail');
    var topbar = document.querySelector('.topbar');
    var ctrls = document.getElementById('topCtrls');
    // 若控制块当前在侧栏里，先挪回顶栏暂存，避免下面 rail.innerHTML='' 把它清掉
    if (ctrls && topbar && ctrls.parentNode === rail) topbar.appendChild(ctrls);
    navTop.innerHTML = '';
    rail.innerHTML = '';
    // 侧栏首行：图标 + WBR Console（与顶栏品牌一致），两种导航位置都显示
    rail.appendChild(el('div', { class: 'rail-brand' }, [
      el('span', { class: 'logo' }),
      el('span', { class: 'rb-text' }, ['WBR Console', el('small', { id: 'railBrandSub', text: 'WBR 控制台' })])
    ]));
    TABS.forEach(function (t) {
      // 【P4 已接】MuJoCo 遥测经 serve.MjTelWriter 落 wbr_tel_mj_*.tsv（plot02 口径），
      //   绘图页与 Isaac 日志同列：指令三线/高度/相位底色可画，Isaac 关节面板缺列自动留空。
      var lock = {};
      navTop.appendChild(el('button', Object.assign({ 'data-tab': t.key, 'aria-pressed': String(S.tab === t.key), text: t.cn }, lock)));
      rail.appendChild(el('button', Object.assign({ 'data-tab': t.key, 'aria-pressed': String(S.tab === t.key), text: t.cn }, lock)));
    });
    $$('[data-tab]').forEach(function (b) {
      b.addEventListener('click', function () { switchTab(b.getAttribute('data-tab')); });
    });
    renderTopCtrls();
    ensureNavTopCtrls();
  }

  function switchTab(key) {
    var was = S.tab;
    S.tab = key; store.set('tab', key);
    $$('[data-tab]').forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-tab') === key)); });
    $$('.page').forEach(function (p) { p.classList.toggle('on', p.getAttribute('data-page') === key); });
    if (key === 'mujoco') { cfxRestore(); cfxFillModels(); cfxFillManifest(); }
    if (key === 'plot') ensurePlot();
    // 原生窗口：离开运行页降下，回到运行页重新贴合
    if (key === 'run') {
      // 【P8 五槽】默认对齐：仿真 run 更换（重启/新启）→ 重置停靠状态并自动重停靠。
      //   旧态 DOCK.on 跨仿真重启残留会让 dockWhenReady 跳过 ⇒ 新窗口停在默认位置错位。
      if (S.sim && S.sim.status === 'running') {
        var _run = S.sim.run || S.sim.id || null;
        if (_run && _run !== lastDockRun) {
          lastDockRun = _run;
          DOCK.on = false; DOCK.cal = null; DOCK.calTries = 0;
          dockWhenReady(false);
        } else {
          dockWhenReady(false);
        }
      }
    } else if (was === 'run') {
      undockWindow(true);
    }
  }

  function setNavPos(pos) {
    S.navpos = pos; store.set('navpos', pos);
    app.setAttribute('data-nav', pos);
    // 顶部：控制块回顶栏（顶栏可见）；左/右侧：控制块进侧栏（顶栏隐藏）
    ensureNavTopCtrls();
    $$('#navposSeg [data-navpos]').forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-navpos') === pos)); });
  }

  /* --------------------------------------------------------------- 配色方案 */
  function applyTheme(theme) {
    S.theme = theme; store.set('theme', theme);
    document.documentElement.setAttribute('data-theme', theme);
    $$('#themeSeg [data-theme]').forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-theme') === theme)); });
    // 绘图页用主题色重绘（网格/坐标/缩略轴），曲线色在 drawPanel 内实时取变量
    if (PLOT.inited) {
      if (PLOT.seg) { try { PLOT.reg.forEach(function (f) { f(); }); } catch (e) { } }
      try { drawOv(); } catch (e) { }
    }
  }
  function bindThemeWatch() {
    try {
      var mq = window.matchMedia('(prefers-color-scheme: dark)');
      var on = function () { if (S.theme === 'system') applyTheme('system'); };
      if (mq.addEventListener) mq.addEventListener('change', on);
      else if (mq.addListener) mq.addListener(on);
    } catch (e) { }
  }

  /* --------------------------------------------------------------- 通道区 */
  var chanEls = {};

  // 【P7F4】Q/E 临时偏航：按住临时覆盖 yaw ±0.5，松开恢复 a/d 设定的基值。
  //   held 位掩码（1=Q, 2=E）支持双键（后按优先）；临时期间 a/d/滑条改的值记为新基值；
  //   yaw 通道禁用（量程退化/离线）时不动作；空格重置会清掉临时态。
  var yawTemp = { base: null, held: 0, dir: 0 };

  function yawSetUI(v) {
    var e = chanEls.ang_vel_yaw; if (!e) return null;
    var spec = CHANNELS.find(function (c) { return c.key === 'ang_vel_yaw'; }); if (!spec) return null;
    v = Math.min(spec.max, Math.max(spec.min, v));
    e.inp.value = v; e.vTxt.textContent = fmt(v, spec.d);
    e.tri.children[0].textContent = 'in ' + fmt(v, spec.d);
    return v;
  }
  function yawTempStart(dir, mask) {
    var e = chanEls.ang_vel_yaw; if (!e || e.inp.disabled) return;
    if (yawTemp.held & mask) return;
    yawTemp.held |= mask;
    if (yawTemp.base === null) yawTemp.base = parseFloat(e.inp.value) || 0;
    yawTemp.dir = dir;
    var v = yawSetUI(0.5 * dir);
    if (v === null) return;
    if (!OFFLINE) POST('/api/cmd', { key: 'ang_vel_yaw', value: v }).catch(function () { });
  }
  function yawTempEnd(mask) {
    if (!(yawTemp.held & mask)) return;
    yawTemp.held &= ~mask;
    if (yawTemp.held) {                    // 另一键仍按住 ⇒ 切到另一方向
      yawTemp.dir = (yawTemp.held & 2) ? 1 : -1;
      var v2 = yawSetUI(0.5 * yawTemp.dir);
      if (v2 !== null && !OFFLINE) POST('/api/cmd', { key: 'ang_vel_yaw', value: v2 }).catch(function () { });
      return;
    }
    if (yawTemp.base === null) return;
    var v = yawSetUI(yawTemp.base);
    yawTemp.base = null;
    if (v !== null && !OFFLINE) POST('/api/cmd', { key: 'ang_vel_yaw', value: v }).catch(function () { });
  }

  function renderChannels() {
    var box = document.getElementById('chans');
    box.innerHTML = ''; chanEls = {};
    CHANNELS.forEach(function (c) {
      var inp = el('input', { type: 'range', min: c.min, max: c.max, step: c.step, value: c.def });
      var vTxt = el('span', { class: 'v', text: fmt(c.def, c.d) });
      var tri = el('div', { class: 'tri' }, [
        el('span', { text: 'in ' + fmt(c.def, c.d) }),
        el('span', { text: 'tgt ' + fmt(c.def, c.d) }),
        el('span', { class: 'a', text: 'act ' + fmt(c.def, c.d) })
      ]);
      var top = el('div', { class: 'chan-top' }, [
        el('b', { text: c.label }), el('span', { class: 'u', text: c.unit }), vTxt
      ]);
      // 指令范围贴在滑轨两端（当前加载模型的实际量程，随模式/模型动态更新）
      var rmin = el('span', { class: 'rng-lo', text: fmt(c.min, c.d) });
      var rmax = el('span', { class: 'rng-hi', text: fmt(c.max, c.d) });
      var slider = el('div', { class: 'slide' }, [rmin, inp, rmax]);
      var wrap = el('div', { class: 'chan', 'data-key': c.key }, [top, slider, tri]);
      box.appendChild(wrap);
      chanEls[c.key] = { wrap: wrap, inp: inp, vTxt: vTxt, tri: tri, spec: c, rmin: rmin, rmax: rmax };

      var sendT = null;
      inp.addEventListener('input', function () {
        var v = parseFloat(inp.value);
        vTxt.textContent = fmt(v, c.d);
        tri.children[0].textContent = 'in ' + fmt(v, c.d);
        if (c.key === 'ang_vel_yaw' && yawTemp.base !== null) yawTemp.base = v;   // 临时偏航期间改滑条 ⇒ 记为新基值
        if (OFFLINE) { tri.children[1].textContent = 'tgt ' + fmt(v * .96, c.d); tri.children[2].textContent = 'act ' + fmt(v * .93, c.d); return; }
        clearTimeout(sendT);
        sendT = setTimeout(function () { POST('/api/cmd', { key: c.key, value: v }).catch(function (e) { toast('下发失败：' + e.message, 'bad'); }); }, 70);
      });
    });
    applyModeToChannels();
  }

  function applyModeToChannels() {
    // 通道是否可用＝当前模式量程是否退化（min===max 即该模式下该指令固定，滑块置灰）。
    // jump：双模式（uni）只在两轮（TWOW）可用——后端两轮才打开跳跃机/放开量程；
    //   四轮/起立/趴下、以及单策略之外的过渡都置灰。单策略恒可用。量程由后端 cmdRanges 动态给。
    var modeKey = (S.state && S.state.modeKey) || null;
    CHANNELS.forEach(function (c) {
      var e = chanEls[c.key]; if (!e) return;
      var fixed = (c.min === c.max);
      var off;
      if (c.key === 'jump_height' && isSlotMode()) {
        off = (modeKey !== 'TWOW');          // uni：仅两轮可跳
      } else {
        off = !!(fixed || (c.switchOff && isSlotMode()));
      }
      e.wrap.classList.toggle('off', off);
      e.inp.disabled = off;
    });
  }

  var lastModeKey = null;
  // 【P8 五槽】点击校准的变换持久化：刷新后无需重新点两下（浏览器挪屏/改缩放后再点即更新）
  try {
    var _savedXf = store.get('vp_xform_v2');          // v2：11-04 污染数据弃用换 key
    if (_savedXf && _savedXf.s > 0.2 && _savedXf.s < 4) {
      DOCK.xform = { ok: true, s: _savedXf.s, ox: _savedXf.ox, oy: _savedXf.oy };
    }
  } catch (_) { }
  // 【P8 五槽·点击校准】页面里每次点击：把页面坐标发给后端，后端配对鼠标指针的
  //   真实设备坐标（xwin.pointer），两组即可解出 CSS→设备变换（停靠目标合成的基准）。
  var xformApplied = false;
  document.addEventListener('click', function (e) {
    if (OFFLINE || !e.detail) return;   // e.detail=0＝键盘触发（指针不在点击处）⇒ 不采集，防污染
    try {
      POST('/api/point', { cx: e.clientX, cy: e.clientY }).then(function (j) {
        // 【P8 五槽】点击校准：后端用指针真值解出 CSS→设备变换 ⇒ 喂 DOCK.xform
        //   （与「校准对齐」按钮同一结构，dockWindow ① 路径直接用）——点击即自动校准。
        if (j && j.ok && j.xform) {
          // 每次点击都刷新变换（浏览器挪屏/改缩放后自动跟上）并重贴（精化即重对齐）
          DOCK.xform = { ok: true, s: j.xform.s, ox: j.xform.ox, oy: j.xform.oy };
          DOCK.cal = { x: 0, y: 0 }; DOCK.calTries = 0;
          try { store.set('vp_xform_v2', { s: DOCK.xform.s, ox: DOCK.xform.ox, oy: DOCK.xform.oy }); } catch (_) { }
          if (S.sim && S.sim.status === 'running') { DOCK.on = false; dockWhenReady(false); }
          if (!xformApplied) {
            xformApplied = true;
            toast('视口校准完成（点击采集 s=' + j.xform.s.toFixed(2) + '），已自动对齐', 'ok');
          }
        }
      }).catch(function () { });
    } catch (_) { }
  }, true);
  var lastDockRun = null;
  function updateReadouts(st) {
    if (!st || !st.cmd) return;
    // 【P8 五槽】模式切换 → 高度滑块跟随：量程已随 cmdRanges 动态（后端 _apply_mode 按模式
    //   切 [0.175,0.245]/[0.2,0.35]/事件窗 [0.24,0.30]），这里补"设定值"回中——后端
    //   _snap_cmds_for 在模式切换时已把目标回中（四轮 0.19 / 两轮 0.28），滑块同步之。
    // 【MuJoCo】没有 modeKey，用链槽位当键；首次拿到 state（_had=false）也同步一次，
    //   否则滑条显示的是 localStorage 旧值（与后端实际指令 0.19/0.25 不一致）。
    var _mk = st.chain ? ('chain:' + (st.slot || '')) : (st.modeKey || '');
    if (_mk !== lastModeKey) {
      var _had = lastModeKey !== null;
      lastModeKey = _mk;
      var _h = chanEls['base_height'];
      // 用 **tgt**（后端实际指令，cmd.h[1]）而不是 in（cmd.h[0] 是前端上次下发值，启动时为 0）
      if (_h && st.cmd.h && st.cmd.h.length > 1) {
        var _vin = parseFloat(st.cmd.h[1]);
        if (!isNaN(_vin) && _vin > 0) {
          _h.inp.value = _vin.toFixed(3);
          _h.vTxt.textContent = _vin.toFixed(3);   // 程序化赋值不触发 input 事件
          gui_cmds['base_height'] = _vin;
        }
      }
    }
    // 相机跟随按钮：显示当前开/关（后端 _STATE.camera，默认关闭）
    var camBtn = document.getElementById('vpCamera');
    if (camBtn) {
      var on = !!st.camera;
      camBtn.textContent = '相机跟随：' + (on ? '开' : '关');
      camBtn.setAttribute('aria-pressed', String(on));
      camBtn.classList.toggle('on', on);
    }
    // 用后端当前模式的真实量程刷新滑块上下限与滑轨两侧标签（四轮/两轮量程不同，随模型动态）
    if (st.cmdRanges) {
      CHANNELS.forEach(function (c) {
        var rg = st.cmdRanges[c.key]; if (!rg || rg.length !== 2) return;
        var e = chanEls[c.key]; if (!e) return;
        c.min = rg[0]; c.max = rg[1];
        e.inp.min = rg[0]; e.inp.max = rg[1];
        if (e.rmin) e.rmin.textContent = fmt(c.min, c.d);
        if (e.rmax) e.rmax.textContent = fmt(c.max, c.d);
      });
    }
    // 模式/量程变化 → 通道可用性与范围随之更新（roll 四轮退化置灰、两轮可用等）
    applyModeToChannels();
    var map = { vx: st.cmd.vx, yaw: st.cmd.yaw, h: st.cmd.h, roll: st.cmd.roll };
    CHANNELS.forEach(function (c) {
      var e = chanEls[c.key]; if (!e) return;
      var t = (c.key === 'jump_height') ? st.cmdJump : map[c.short];
      if (!t) return;
      e.tri.children[0].textContent = 'in ' + fmt(t[0], c.d);
      e.tri.children[1].textContent = 'tgt ' + fmt(t[1], c.d);
      e.tri.children[2].textContent = 'act ' + fmt(t[2], c.d);
    });
    // 模式
    var pill = document.getElementById('modeNowPill');
    if (st.switch && st.modeCN) {
      pill.textContent = '模式：' + st.modeCN;
      document.getElementById('btnRise').disabled = (st.modeKey !== 'FOURW');
      document.getElementById('btnLower').disabled = (st.modeKey !== 'TWOW');
    } else {
      pill.textContent = '模式：单策略';
      document.getElementById('btnRise').disabled = true;
      document.getElementById('btnLower').disabled = true;
    }
    renderStatus(st);
  }

  function renderStatus(st) {
    var box = document.getElementById('runStatus');
    var g = (S.sim && S.sim.gpu) || {};
    var items = [];
    if (st && st.hasReadout) {
      items.push(['mode', st.modeCN || '—']);
      items.push(['ph', String(st.phUni != null ? st.phUni : st.ph)]);
      if (st.ctrl != null) items.push(['ctrl', st.ctrl === 1 ? '跳跃网(P8J)' : '行走网(P8W7)']);
      items.push(['ev_up', st.evUp ? '●' : '○']);
    }
    if (st && st.telemetry) {
      items.push(['遥测', st.telemetry.file || '—']);
      items.push(['采样', (st.telemetry.rows || 0) + ' 行']);
    }
    items.push(['GPU', (g.used != null ? g.used + '/' + g.total + 'G' : '—')]);
    items.push(['仿真', S.sim.status === 'running' ? ('pid ' + S.sim.pid) : '未启动']);
    items.push(['视口校准', (DOCK.xform && DOCK.xform.ok)
      ? ('已完成（s=' + DOCK.xform.s.toFixed(2) + '）')
      : '未完成——在页面上任意点击两处不同按钮即可']);
    // 状态条同样避免无谓重排（遥测行数每拍变，但布局不该跟着抖）
    var sig = items.map(function (it) { return it[0] + '=' + it[1]; }).join('|');
    if (box._sig !== sig) {
      box.innerHTML = items.map(function (it) { return '<span>' + it[0] + ' <b>' + it[1] + '</b></span>'; }).join('');
      box._sig = sig;
    }
    // 【S3 / A+B】MuJoCo 五槽链：指令回执条 + 动作按钮按槽位解禁
    var bar = document.getElementById('mjChainBar');
    // 用桥上报的**实际引擎**（不是用户选择 S.engine）：从配置页一键启动链时用户没点引擎分段，
    // S.engine 仍是 isaac，若用它会让 chainOn=false、按钮不解禁、时序条不显示。
    var mjNow = (st && st.engine === 'mujoco') || displayEngine() === 'mujoco';
    var chainOn = (mjNow && st && st.chain);
    var allowed = (chainOn && st.allowed) || [];
    var isChainSlot = function (a) { return chainOn && allowed.indexOf(a) >= 0; };

    // 动作按钮：链模式按守卫表解禁；单槽模式保留「装填跳」原逻辑，其余置灰
    var btnMap = [
      ['#btnRise', 'rise', 'MuJoCo 单策略不支持起立（链模式四轮槽可点）'],
      ['#btnLower', 'lower', 'MuJoCo 单策略不支持趴下（链模式两轮槽可点）'],
      ['[data-act="armjump"]', 'armjump', '当前槽位无跳跃通道'],
      ['#btnMjFour', 'four', '当前槽位不能回四轮'],
      ['#tglAuto', 'auto', '单策略不支持连跳']
    ];
    btnMap.forEach(function (b) {
      var n = document.querySelector(b[0]); if (!n) return;
      var can = chainOn
        ? (b[1] === 'armjump' ? (st.canJump || isChainSlot('jump')) : isChainSlot(b[1]))
        : (b[1] === 'armjump' && mjNow && st && st.canJump);
      if (can) {
        n.removeAttribute('data-mj-locked');
        n.removeAttribute('data-mj-why');
        if (n.tagName === 'BUTTON' || n.tagName === 'INPUT') n.disabled = false;
      } else {
        n.setAttribute('data-mj-locked', '');
        n.setAttribute('data-mj-why', b[2]);
      }
    });
    // 槽位直达按钮：链模式显示，且按守卫表解禁（不在 allowed 里就置灰）
    ['#btnMjTwo', '#btnMjFour'].forEach(function (id) {
      var n = document.querySelector(id); if (!n) return;
      n.hidden = !chainOn;
      if (!chainOn) return;
      var req = id === '#btnMjTwo' ? 'two' : 'four';
      if (allowed.indexOf(req) >= 0) { n.removeAttribute('data-mj-locked'); n.removeAttribute('data-mj-why'); }
      else { n.setAttribute('data-mj-locked', ''); n.setAttribute('data-mj-why', '当前槽位不能' + (req === 'two' ? '回两轮' : '回四轮')); }
    });

    if (bar) {
      if (chainOn) {
        var evs = (st.marks || []).slice(-6).map(function (e) {
          if (!e || typeof e !== 'object') return '';
          if (Array.isArray(e)) {                       // 旧格式 [name, t, z]
            return '▶ ' + e[0] + ' @' + e[1] + 's z=' + e[2];
          }
          if (e.event === 'request') {
            return (e.accepted ? '✅' : '⛔') + ' 指令「' + (e.label || e.req) + '」' +
                   (e.reason ? '：' + e.reason : '');
          }
          if (e.event === 'applied') {
            return '▶ 执行「' + (e.label || e.req) + '」→ ' + (e.slot || '?');
          }
          if (e.event === 'terminated') {
            return '⛔ 终止：' + (e.why || '?');
          }
          return '▶ ' + (e.slot || '?') + ' @' + (e.t != null ? e.t : '?') +
                 's z=' + (e.z != null ? e.z : '?');
        }).filter(Boolean).join('<br>');
        var lr = st.lastRequest;
        var html = '链槽位 <b>' + (st.slot || '—') + '</b> · 相位 <b>' + st.phase + '</b> · ' +
          '可下指令 <b>' + (allowed.join(' / ') || '—') + '</b> · ' +
          '指令 <b>' + (st.nRequests || 0) + '</b> 条（拒绝 ' + (st.nRejected || 0) + '） · ' +
          (st.terminated ? '<b style="color:#c33">终止：' + st.terminated + '</b>' : '运行中，等指令') +
          (lr ? '<br><span class="muted">上一条：' + (lr.accepted ? '✅' : '⛔') + ' ' +
                (lr.label || lr.req) + ' — ' + (lr.reason || '') + '</span>' : '') +
          '<br><span class="muted">' + evs + '</span>';
        // 内容没变就不写 DOM（每 600ms 轮询时避免无谓重排 → 不触发 ResizeObserver/重贴窗口）
        if (bar._sig !== html) { bar.innerHTML = html; bar._sig = html; }
        bar.hidden = false;
      } else {
        if (!bar.hidden) { bar.hidden = true; bar.innerHTML = ''; bar._sig = null; }
      }
    }
  }

  function renderSimState() {
    var dot = document.getElementById('simDot');
    var txt = document.getElementById('simStateText');
    dot.className = 'dot';
    if (OFFLINE) { txt.textContent = '离线预览（无后端）'; return; }
    if (S.sim.status === 'running') { dot.classList.add('live'); txt.textContent = '仿真运行中 · pid ' + S.sim.pid; }
    else { txt.textContent = '仿真未启动'; }
    var vp = document.getElementById('viewport');
    vp.classList.toggle('docked', S.sim.status === 'running');
    document.getElementById('btnStart').disabled = S.sim.status === 'running';
    document.getElementById('btnStart2').disabled = S.sim.status === 'running';
    document.getElementById('btnStop').disabled = S.sim.status !== 'running';
    applyEngineUI();                       // 【Sim2Sim P2】徽标/分段锁定随运行态刷新
    if (S.sim.status === 'running') {
      if (S.tab === 'run') {
      // 【P8 五槽】仿真 run 更换（重启/新启）→ 重置停靠状态并重停靠
      //   （旧 DOCK.on 残留会让 dockWhenReady 跳过 ⇒ 新窗口停在默认位置错位）
      var _run = S.sim && S.sim.run;
      if (_run && _run !== lastDockRun) {
        lastDockRun = _run;
        DOCK.on = false; DOCK.cal = null; DOCK.calTries = 0;
      }
      dockWhenReady(false);
    }
    } else {
      DOCK.on = false;
      clearInterval(DOCK.timer);
      setVpStatus('仿真视口 · 自适应填充');
    }
  }

  /* ------------------------------------------------------------- 抽屉 */
  function renderDrawer() {
    var logBox = document.getElementById('drawerLog');
    logBox.innerHTML = '';
    if (OFFLINE && MOCK) {
      MOCK.logLines.forEach(function (l) {
        var span = el('span', { class: l.level === 'ok' ? 'ok' : l.level === 'warn' ? 'wn' : '' });
        span.innerHTML = '<span class="k">' + l.t + '</span>  ' + l.text;
        logBox.appendChild(span); logBox.appendChild(document.createTextNode('\n'));
      });
    } else {
      logBox.appendChild(el('span', { text: '（运行记录见下方列表；点「刷新」拉取最近一次 console.log）' }));
    }
    $$('#drawerTabs button').forEach(function (b) {
      b.addEventListener('click', function () {
        var k = b.getAttribute('data-drawer');
        $$('#drawerTabs button').forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
        $$('[data-dbody]').forEach(function (d) { d.hidden = d.getAttribute('data-dbody') !== k; });
        if (k === 'log') loadRuns();
      });
    });
  }

  function loadRuns() {
    var box = document.getElementById('drawerLog');
    GET('/api/runs').then(function (j) {
      S.runs = j.runs || [];
      box.innerHTML = '';
      if (!S.runs.length) { box.appendChild(el('span', { text: '还没有运行记录' })); return; }
      S.runs.slice(0, 12).forEach(function (r) {
        var row = el('div', { class: 'item', style: 'margin-bottom:.25rem' }, [
          el('span', { class: 'nm', text: r.id }),
          el('span', { class: 'meta', text: r.mode + ' · ' + r.when + ' · ' + r.sizeKB + 'KB' }),
          el('button', { text: '查看', onclick: function () { showRunLog(r.id); } })
        ]);
        box.appendChild(row);
      });
    }).catch(function () { });
  }

  function showRunLog(id) {
    GET('/api/runlog?id=' + encodeURIComponent(id)).then(function (j) {
      var box = document.getElementById('drawerConsole');
      box.textContent = j.text || '（空）';
      $$('#drawerTabs button').forEach(function (x) { x.setAttribute('aria-pressed', String(x.getAttribute('data-drawer') === 'console')); });
      $$('[data-dbody]').forEach(function (d) { d.hidden = d.getAttribute('data-dbody') !== 'console'; });
    }).catch(function (e) { toast('读取失败：' + e.message, 'bad'); });
  }

  /* ------------------------------------------------------------- 模型页 */
  function slotObj(key) { return S.slots[key]; }

  function enrichSlots() {
    SLOT_ORDER.forEach(function (k) {
      var s = S.slots[k];
      if (!s || !s.path || s.obs != null) return;
      var c = S.checkpoints.find(function (x) { return x.path === s.path; });
      if (c) { s.obs = c.obs; s.act = c.act; s.model = c.model; s.ckpt = c.ckpt; }
    });
  }

  function renderChain() {
    var chain = document.getElementById('chain');
    chain.innerHTML = '';
    SLOT_ORDER.forEach(function (key, i) {
      var s = S.slots[key] || {};
      var obsOk = (s.obs == null) ? false : (s.obs === SLOT_OBS[key]);
      var badges = s.script
        ? [el('span', { class: 'badge ok', text: '无需权重' })]
        : [
          el('span', { class: 'badge ' + (obsOk ? 'ok' : 'warn'), text: 'obs ' + (s.obs == null ? '?' : s.obs) + (obsOk ? ' ✓' : ' ✗') }),
          el('span', { class: 'badge', text: 'act ' + (s.act == null ? '?' : s.act) })
        ];
      var slot = el('div', { class: 'slot', 'data-slot': key }, [
        el('div', { class: 'stage', text: SLOT_STAGE[key] }),
        el('div', { class: 'model', text: s.model || (s.script ? SCRIPT_SLOT_LABEL : '（未选择）'), title: s.path || '' }),
        el('div', { class: 'ckpt', text: s.ckpt || '—', title: s.path || '' }),
        el('div', { class: 'badges' }, badges),
        el('div', { class: 'slotbtns' }, [
          el('button', { text: s.script ? '换成模型' : '更换', onclick: function () { openPicker(key); } }),
          el('button', { text: '☆ 收藏', onclick: function () { favoriteSlot(key); } })
        ])
      ]);
      chain.appendChild(slot);
      if (i < SLOT_ORDER.length - 1) chain.appendChild(el('div', { class: 'arrow', text: '→' }));
    });
  }

  function renderChainParams() {
    var box = document.getElementById('chainParams');
    if (!box) return;
    box.innerHTML = '';
    var p = S.params || {};
    var val = function (k) { return (p[k] != null) ? p[k] : PARAM_DEFAULTS[k]; };
    var mode = val('standup_mode') === 'trad' ? 'trad' : 'p7s';
    var defOf = {}; PARAM_DEFS.forEach(function (d) { defOf[d.key] = d; });

    function makeInput(def) {
      var inp;
      if (def.type === 'select') {
        inp = el('select', { class: 'input pin sel' });
        def.opts.forEach(function (o) { inp.appendChild(el('option', { value: o[0], text: o[1] })); });
        inp.value = val(def.key);
        inp.addEventListener('change', function () { S.params[def.key] = inp.value; persistParams(); renderChainParams(); runPreflight(true); });
      } else {
        inp = el('input', { type: 'number', class: 'input pin', min: def.min, max: def.max, step: def.step, value: val(def.key) });
        inp.addEventListener('change', function () {
          var v = parseFloat(inp.value);
          if (!isFinite(v)) { inp.value = val(def.key); return; }
          v = Math.min(def.max, Math.max(def.min, v));
          inp.value = v; S.params[def.key] = v; persistParams(); runPreflight(true);
        });
      }
      return inp;
    }

    PARAM_GROUPS.forEach(function (grp) {
      var dim = grp.when && grp.when !== mode;   // 非当前起立方式的组置灰（保留位置）
      var group = el('div', { class: 'pgroup' + (dim ? ' dim' : '') }, [
        el('div', { class: 'pgroup-h' }, [
          el('b', { text: grp.title }),
          el('span', { class: 'muted', text: '  ' + (grp.hint || '') })
        ])
      ]);
      var row = el('div', { class: 'params' });
      grp.keys.forEach(function (k) {
        var def = defOf[k]; if (!def) return;
        var inp = makeInput(def);
        if (dim) inp.disabled = true;
        row.appendChild(el('label', { class: 'param edit', title: grp.title + '（' + (grp.hint || '') + '；重启生效）' }, [
          document.createTextNode(def.label + ' '), inp, el('span', { class: 'u', text: def.unit || '' })
        ]));
      });
      group.appendChild(row);
      box.appendChild(group);
    });
    var badge = document.getElementById('chainModeBadge'); if (badge) badge.textContent = '按生效位置';
  }

  function persistParams() {
    var out = {};
    PARAM_DEFS.forEach(function (d) { out[d.key] = S.params[d.key] != null ? S.params[d.key] : PARAM_DEFAULTS[d.key]; });
    POST('/api/settings', { params: out }).catch(function () { });
  }

  function renderPreflight(items) {
    // 【2026-10-29】地形任务：preflight 里会多一行 "地形：xxx.npz"，直接显示会加载哪张图
    var box = document.getElementById('preflight');
    box.innerHTML = '';
    (items || []).forEach(function (p) {
      var cls = p.level === 'ok' ? 'ok' : p.level === 'warn' ? 'wn' : p.level === 'bad' ? 'bd' : 'muted';
      var mark = p.level === 'ok' ? '✓' : p.level === 'warn' ? '⚠' : p.level === 'bad' ? '✗' : '·';
      box.appendChild(el('span', { class: cls }, [el('b', { text: mark }), document.createTextNode(' ' + p.text)]));
    });
  }

  function buildSpec() {
    var params = {};
    PARAM_DEFS.forEach(function (d) { params[d.key] = (S.params && S.params[d.key] != null) ? S.params[d.key] : PARAM_DEFAULTS[d.key]; });
    if (isSlotMode()) {
      var slots = {};
      SLOT_ORDER.forEach(function (k) {
        var s = S.slots[k];
        slots[k] = s ? (s.script ? { type: 'script' } : s.path) : null;
      });
      // 起立槽＝相位脚本时强制 standup_mode=script（env 内 RiseScript 直走，不走"权重缺失回退"分支）
      if (S.slots.standup && S.slots.standup.script) params.standup_mode = 'script';
      return { mode: S.mode, slots: slots, params: params };
    }
    var sp = S.single;
    // 【2026-10-29】地形任务（wbr_terrain / wbr_demo / wbr_combo / wbr_elements）**不传
    //   from_run_config**：历史 run 的 config 会把地形几何（mesh_type/高度图/行列数/垂直分辨率）
    //   回填成平地，控制台就只剩一块平地面。奖励/指令参数本来就与 W 线一致，回填没有意义。
    var _terrainTask = /^wbr_(terrain|demo|combo|elements)$/.test(String(sp.task || ''));
    return {
      mode: 'single',
      single: {
        task: sp.task, load_run: sp.run || null, checkpoint: sp.checkpoint,
        from_run_config: _terrainTask ? null : (sp.from_run_config || null),
        ckpt_path: sp.path || null,
        robots: Math.max(1, Math.min(3, parseInt(sp.robots, 10) || 3))
      },
      params: params
    };
  }

  function runPreflight(silent) {
    var spec = (S.engine === 'mujoco') ? buildMujocoSpec() : buildSpec();
    POST('/api/preflight', { spec: spec }).then(function (j) {
      renderPreflight(j.items);
      if (!silent) {
        var bad = (j.items || []).filter(function (i) { return i.level === 'bad'; });
        toast(bad.length ? ('预检有 ' + bad.length + ' 项不通过') : '预检通过', bad.length ? 'bad' : 'ok');
      }
    }).catch(function (e) { toast('预检失败：' + e.message, 'bad'); });
  }

  function renderScenarios() {
    var box = document.getElementById('scenarioList');
    box.innerHTML = '';
    if (!S.scenarios.length) { box.appendChild(el('div', { class: 'muted', style: 'font-size:.72rem', text: '还没有场景，点右上「保存场景」' })); return; }
    S.scenarios.forEach(function (sc) {
      var meta = (sc.mode === 'single' ? '单策略' : '双模式') + ' · ' + (sc.when || '');
      box.appendChild(el('div', {
        class: 'item', title: '双击加载场景：' + sc.name,
        ondblclick: function () { loadScenario(sc); }
      }, [
        el('span', { class: 'nm', text: sc.name, title: sc.name }),
        el('span', { class: 'meta', text: meta }),
        el('span', { class: 'acts' }, [
          iconBtn('pen', '覆写到本条', function () { openSaveScenario(sc); }),
          iconBtn('del', '删除本条', function () { confirmDeleteScenario(sc); })
        ])
      ]));
    });
  }

  function renderSaved() {
    var box = document.getElementById('savedList');
    box.innerHTML = '';
    if (!S.saved.length) { box.appendChild(el('div', { class: 'muted', style: 'font-size:.72rem', text: '还没有固化存档' })); return; }
    S.saved.forEach(function (s) {
      box.appendChild(el('div', {
        class: 'item', title: '双击加载到单策略：' + s.name,
        ondblclick: function () { loadSavedToSingle(s); }
      }, [
        el('span', { class: 'nm', text: s.name, title: s.source }),
        el('span', { class: 'meta', text: (s.obs || '?') + '·' + (s.act || '?') + ' · ' + s.sizeMB + 'MB · ' + s.when }),
        el('span', { class: 'acts' }, [
          iconBtn('pen', '覆写到本条', function () { openSolidifyOverwrite(s); }),
          iconBtn('del', '删除本条', function () { confirmDeleteSaved(s); })
        ])
      ]));
    });
  }

  // 从当前 run 目录 + task 维度列出 checkpoint → 填充 checkpoint 下拉（让下拉真正可选，不再单值）
  function renderSingleCkptOptions() {
    var sp = S.single;
    var ckSel = document.getElementById('singleCkpt');
    ckSel.innerHTML = '';
    var want = TASK_OBS[sp.task];
    var inRun = (S.checkpoints || []).filter(function (c) {
      if (sp.run && c.path.indexOf(sp.run + '/') !== 0) return false;
      if (want != null && c.obs !== want) return false;
      return true;
    }).sort(function (a, b) { return (b.mtime || 0) - (a.mtime || 0); });
    if (!inRun.length) {
      ckSel.appendChild(el('option', { text: sp.checkpoint != null ? ('model_' + sp.checkpoint) : '—', value: sp.checkpoint != null ? String(sp.checkpoint) : '' }));
      return;
    }
    var seen = {};
    inRun.forEach(function (c) {
      var num = parseInt((c.ckpt.match(/(\d+)/) || [0, 0])[1], 10);
      if (seen[num]) return; seen[num] = true;
      ckSel.appendChild(el('option', { text: c.ckpt, value: String(num) }));
    });
    if (sp.checkpoint != null) ckSel.value = String(sp.checkpoint);
  }

  function renderSingle() {
    var sp = S.single;
    document.getElementById('singleTask').value = sp.task || 'wbr_flat';
    document.getElementById('singleRobots').value = String(sp.robots || 3);
    var runSel = document.getElementById('singleRun');
    runSel.innerHTML = '';
    var runName = sp.run ? String(sp.run).split('/').pop() : '';
    if (runName) runSel.appendChild(el('option', { text: runName, value: sp.run }));
    else runSel.appendChild(el('option', { text: '（从检查点库选择）', value: '' }));
    renderSingleCkptOptions();
    var b = document.getElementById('singleBadge2');
    var want = TASK_OBS[sp.task];
    var mismatch = (sp.obs != null && want != null && sp.obs !== want);
    if (sp.obs == null) { b.textContent = 'obs —'; b.className = 'badge warn'; }
    else if (mismatch) { b.textContent = 'obs ' + sp.obs + ' ✗ 需 ' + want + '（task 不符）'; b.className = 'badge bad'; }
    else { b.textContent = 'obs ' + sp.obs + ' ✓'; b.className = 'badge ok'; }
  }

  function applyModeToModelPage() {
    var isSingle = S.mode === 'single';
    document.getElementById('modelModeName').textContent = MODE_LABEL[S.mode] || MODE_LABEL.uni;
    document.getElementById('runModePill').textContent = MODE_LABEL[S.mode] || MODE_LABEL.uni;
    $$('.cfg-card').forEach(function (c) { c.setAttribute('data-current', String(c.getAttribute('data-mode') === S.mode)); });
    $$('.cfg-use').forEach(function (b) { b.textContent = (b.getAttribute('data-mode') === S.mode) ? '✓ 当前使用' : '设为当前'; });
    var sw = SLOT_ORDER.map(function (k) {
      var s = S.slots[k];
      return (s && (s.path || s.script)) ? s.model : '—';
    }).join(' / ');
    var mcUni = document.getElementById('mcUniMeta'); if (mcUni) mcUni.textContent = sw;
    var smSel = document.getElementById('standupModeSel');
    if (smSel) smSel.value = (S.params && S.params.standup_mode === 'trad') ? 'trad' : 'p7s';
    document.getElementById('mcSingleMeta').textContent = S.single.path
      ? (S.single.path.split('/').slice(-2).join('/') + (S.single.obs != null ? ' · obs ' + S.single.obs : ''))
      : '未选择 checkpoint';
    renderChain(); renderChainParams(); applyModeToChannels();
  }

  /* --------------------------------------------------------- 检查点选择器 */
  function openPicker(key) {
    S.pickerSlot = key;
    var reqObs = key === 'single' ? (S.single.obs || null) : SLOT_OBS[key];
    document.getElementById('pickerTitle').textContent = key === 'single'
      ? '选择单策略 checkpoint' : ('为「' + SLOT_STAGE[key] + '」选择 checkpoint');
    document.getElementById('pickerReq').textContent = key === 'single'
      ? 'act 6 · obs 161/164/173 均可（按维度自动定 task）'
      : ('需要 obs ' + (reqObs == null ? '—' : reqObs) + ' · act 6');
    document.getElementById('pickerSearch').value = '';
    document.getElementById('pickerSource').value = 'all';
    document.getElementById('pickerFavOnly').checked = false;
    ensureCheckpoints().then(function () { renderPicker(); document.getElementById('dlgPicker').showModal(); });
  }

  function ensureCheckpoints() {
    if (S.checkpoints.length) return Promise.resolve(S.checkpoints);
    return GET('/api/models').then(function (j) { S.checkpoints = j.checkpoints || []; return S.checkpoints; })
      .catch(function () { return []; });
  }

  function renderPicker() {
    var key = S.pickerSlot;
    if (!key) return;
    // 单策略：不限制 obs（选谁就按它的维度自动定 task）；四槽：必须匹配槽位 obs。
    var reqObs = key === 'single' ? null : SLOT_OBS[key];
    var q = document.getElementById('pickerSearch').value.trim().toLowerCase();
    var src = document.getElementById('pickerSource').value;
    var favOnly = document.getElementById('pickerFavOnly').checked;
    var matchOnly = document.getElementById('pickerMatchOnly').checked;
    var favSet = {};
    S.saved.forEach(function (s) { favSet[s.source] = true; });
    var body = document.getElementById('pickerBody');
    body.innerHTML = '';

    var rows = S.checkpoints.filter(function (c) {
      if (src !== 'all' && c.source !== src) return false;
      if (favOnly && !favSet[c.path]) return false;
      if (matchOnly && reqObs != null && !(c.obs === reqObs && c.act === 6)) return false;
      if (q && (c.model + ' ' + c.path).toLowerCase().indexOf(q) < 0) return false;
      return true;
    });
    if (!rows.length) { body.appendChild(el('tr', {}, [el('td', { colspan: '7', class: 'muted', text: '没有匹配的 checkpoint（后端不可达时列表为空）' })])); return; }

    rows.slice(0, 400).forEach(function (c) {
      // 单策略：act=6 且 obs∈{161,164,173}（能映射到 task）即可选；四槽：须匹配槽位 obs。
      var ok = (key === 'single')
        ? (c.act === 6 && TASK_OBS_INV[c.obs] != null)
        : ((reqObs == null) ? true : (c.obs === reqObs && c.act === 6));
      var tr = el('tr', { class: ok ? 'pickable' : 'disabled' }, [
        el('td', {}, [document.createTextNode(c.model), favSet[c.path] ? el('span', { class: 'badge pri', style: 'margin-left:.35rem', text: '★' }) : null]),
        el('td', { class: 'mono', text: c.ckpt }),
        el('td', { text: (c.obs == null ? '?' : c.obs) + ' · ' + (c.act == null ? '?' : c.act) }),
        el('td', { text: c.sizeMB + 'MB' }),
        el('td', { text: c.source }),
        el('td', { text: new Date(c.mtime * 1000).toLocaleDateString() }),
        el('td', { class: 'path', title: c.path, text: c.path })
      ]);
      if (ok) tr.addEventListener('click', function () { applyPick(c); document.getElementById('dlgPicker').close(); });
      body.appendChild(tr);
    });
  }

  // 由 checkpoint 的观测维度推断承载它的 task：
  //   164=wbr_flat4(四轮) / 173=wbr_switch(起立·趴下) 一一对应；
  //   161 现在对应**三个**任务（wbr_flat 平地 / wbr_terrain 起伏地形 / wbr_jump 跳跃，观测同维）
  //   ⇒ 先看路径里有没有任务名线索，再退回 wbr_flat。
  function taskForObs(obs, path) {
    var p = String(path || '');
    if (/wbr_terrain/.test(p)) return 'wbr_terrain';
    if (/wbr_jump|P8J/.test(p)) return 'wbr_jump';
    if (/wbr_flat4/.test(p)) return 'wbr_flat4';
    if (/wbr_switch/.test(p)) return 'wbr_switch';
    if (obs === 164) return 'wbr_flat4';
    if (obs === 173) return 'wbr_switch';
    return 'wbr_flat';
  }

  function applyPick(c) {
    var key = S.pickerSlot;
    if (key === 'single') {
      var dir = c.path.split('/').slice(0, -1).join('/');
      var num = parseInt((c.ckpt.match(/(\d+)/) || [0, 0])[1], 10);
      S.single = {
        task: taskForObs(c.obs, c.path),
        run: dir, checkpoint: num, from_run_config: dir, path: c.path, obs: c.obs, act: c.act
      };
      renderSingle();
    } else {
      S.slots[key] = { path: c.path, model: c.model, ckpt: c.ckpt, obs: c.obs, act: c.act };
      renderChain(); persistSlots();
    }
    runPreflight(true);
    toast((key === 'single' ? '单策略' : SLOT_STAGE[key]) + ' ← ' + c.model + ' / ' + c.ckpt, 'ok');
  }

  function favoriteSlot(key) {
    var s = S.slots[key];
    if (!s || !s.path) { toast('该槽还没有 checkpoint', 'warn'); return; }
    solidify(s.path, s.obs, s.act);
  }
  function favoriteCurrent() {
    if (S.mode === 'single') { if (S.single.path) solidify(S.single.path, S.single.obs, S.single.act); else toast('单策略还没选 checkpoint', 'warn'); return; }
    favoriteSlot('standup');
  }
  function solidify(path, obs, act) {
    POST('/api/saved', { path: path, obs: obs, act: act }).then(function (j) {
      toast('已固化：' + j.meta.name, 'ok'); loadSaved();
    }).catch(function (e) { toast('固化失败：' + e.message, 'bad'); });
  }

  /* ------------------------------------------------------------- 绘图页 */
  // 复用 plot02 的 Index/read_window/read_overview（后端 bench_plot 已封装），前端移植它的
  // 18 子图绘制/缩略轴/导出。只把配色接到主题变量，使浅色/深色都成立。
  var PLOT = { inited: false, name: '', meta: null, ov: null, seg: null, reg: [], drag: null };

  var KIND = {
    'in': { color: '#94a3b8', w: 1.3, a: 0.95, dash: [2, 3] },
    'tgt': { color: '#f59e0b', w: 5.0, a: 0.38, dash: [] },
    'act': { color: '#2563eb', w: 1.7, a: 1.0, dash: [] },
    'alt': { color: '#0f766e', w: 1.7, a: 1.0, dash: [] },
    'alt2': { color: '#7c3aed', w: 1.7, a: 1.0, dash: [] },
    'jump': { color: '#6b7280', w: 1.5, a: 0.95, dash: [5, 3] },
    'wheel': { color: '#7f1d1d', w: 2.4, a: 1.0, dash: [] },
    'air': { color: '#ea580c', w: 1.7, a: 1.0, dash: [] }
  };

  function themeColors() {
    var cs = getComputedStyle(document.documentElement);
    var g = function (k, d) { return (cs.getPropertyValue(k) || '').trim() || d; };
    return { fg: g('--foreground', '#111827'), muted: g('--muted-foreground', '#64748b'),
             border: g('--border', '#e2e8f0'), card: g('--card', '#ffffff') };
  }

  function isDarkTheme() {
    var t = document.documentElement.getAttribute('data-theme');
    if (t === 'dark') return true;
    if (t === 'light') return false;
    try { return window.matchMedia('(prefers-color-scheme: dark)').matches; } catch (e) { return false; }
  }
  // 深色下把曲线颜色提亮，保证在深底上可读
  var KIND_DARK = {
    'in': '#94a3b8', 'tgt': '#fbbf24', 'act': '#60a5fa', 'alt': '#2dd4bf',
    'alt2': '#a78bfa', 'jump': '#9ca3af', 'wheel': '#fb7185', 'air': '#fb923c'
  };
  // 模式色带在深色下用低饱和深底，避免刺眼
  var BAND_DARK = {
    '#dbeafe': '#1e3350', '#d1fae5': '#12352b', '#fef3c7': '#3a3016', '#ede9fe': '#2c2547'
  };

  function ensurePlot() {
    if (PLOT.inited) { return; }
    PLOT.inited = true;
    var sel = document.getElementById('plotLog');
    sel.addEventListener('change', function () { openPlot(sel.value); });
    document.getElementById('plotDraw').addEventListener('click', loadWindow);
    document.getElementById('plotFollow').addEventListener('change', function (e) {
      if (e.target.checked) tickFollow(); else if (PLOT.meta) openPlot(PLOT.name, true);
    });    ['plotT0', 'plotT1', 'plotWid'].forEach(function (id) {
      var inp = document.getElementById(id);
      inp.addEventListener('input', function () { markDirty(true); });
      inp.addEventListener('keydown', function (e) { if (e.key === 'Enter') loadWindow(); });
    });
    document.getElementById('plotT0').addEventListener('change', function () { setWin(+document.getElementById('plotT0').value, +document.getElementById('plotT1').value); });
    document.getElementById('plotT1').addEventListener('change', function () { setWin(+document.getElementById('plotT0').value, +document.getElementById('plotT1').value); });
    document.getElementById('plotWid').addEventListener('change', function () { setWin(+document.getElementById('plotT1').value - +document.getElementById('plotWid').value, +document.getElementById('plotT1').value); });
    $$('[data-quick]').forEach(function (b) {
      b.addEventListener('click', function () {
        var q = b.getAttribute('data-quick');
        if (q === 'all') { if (PLOT.ov) setWin(PLOT.ov.t0, PLOT.ov.t1); }
        else setWin(+document.getElementById('plotT1').value - parseFloat(q), +document.getElementById('plotT1').value);
        loadWindow();
      });
    });
    document.getElementById('plotSavePage').addEventListener('click', saveComposite);
    document.getElementById('plotSaveCells').addEventListener('click', saveAll);
    document.getElementById('plotRefresh').addEventListener('click', function () { loadPlotList({ refresh: true }); });
    bindOv();
    addEventListener('resize', function () { PLOT.reg.forEach(function (f) { f(); }); drawOv(); });
    updateCfgChip();
    ensureFollowTimer();
    loadPlotList();
  }

  function loadPlotList(opts) {
    opts = opts || {};
    GET('/api/plot/list').then(function (j) {
      var sel = document.getElementById('plotLog');
      var logs = j.logs || [];
      sel.innerHTML = '';
      if (!logs.length) { sel.appendChild(el('option', { text: '（wbr_play_data 里还没有遥测）', value: '' })); setPlotStatus('目录里没有遥测文件，先跑一次仿真采数据'); return; }
      logs.forEach(function (l) {
        var tag = l.kind === 'switch' ? ' · 双模式' : ' · 单策略';
        if (l.sample) tag += ' · 样例';
        sel.appendChild(el('option', { value: l.name, text: l.name + tag }));
      });
      var names = logs.map(function (l) { return l.name; });
      var cur = PLOT.name;
      // 刷新按钮：只重拉列表；当前选择仍在列表里就保留（不重置时间窗），否则回退到最近有数据的
      if (opts.refresh && cur && names.indexOf(cur) >= 0) {
        sel.value = cur;
        toast('文件列表已刷新（共 ' + names.length + ' 个）', 'ok');
        return;
      }
      // 首次进入/切页：当前文件仍存在就保留，否则选「最近一个有数据行」的文件（跳过只有表头的空文件）
      if (!opts.refresh && cur && names.indexOf(cur) >= 0) {
        sel.value = cur;
        return;
      }
      pickLatestWithData(names).then(function (pick) {
        sel.value = pick;
        openPlot(pick);
      });
    }).catch(function () { setPlotStatus('后端不可达，绘图不可用'); });
  }

  // 从新→旧依次探测 meta.n，返回最近一个有数据行的文件名；全空则退回第一个
  function pickLatestWithData(names) {
    var i = 0;
    function step() {
      if (i >= names.length) return Promise.resolve(names[0]);
      var nm = names[i++];
      return GET('/api/plot/meta?name=' + encodeURIComponent(nm)).then(function (m) {
        return (m && m.n) ? nm : step();
      }).catch(function () { return step(); });
    }
    return step();
  }

  function setPlotStatus(t) { document.getElementById('plotSeg').textContent = t; }

  function openPlot(name, keepWin) {
    if (!name) return;
    PLOT.name = name;
    document.getElementById('plotLog').value = name;
    GET('/api/plot/meta?name=' + encodeURIComponent(name)).then(function (m) {
      if (!m || !m.n) { setPlotStatus('该次运行没采到数据（' + name + '），已保留在列表里但无法绘图'); return; }
      PLOT.meta = m;
      document.getElementById('plotKind').textContent = m.kind === 'switch' ? '双模式' : '单策略';
      document.getElementById('plotKind').className = 'kindbadge ' + (m.kind === 'switch' ? 'sw' : 'sg');
      updateCfgChip();
      return GET('/api/plot/overview?name=' + encodeURIComponent(name)).then(function (ov) {
        PLOT.ov = (ov && ov.zmin) ? ov : { t0: m.t0, t1: m.t1 };
        if (!keepWin) {
          var follow = document.getElementById('plotFollow').checked;
          var b = follow ? m.t1 : Math.min(m.t1, m.t0 + 10);
          var a = Math.max(m.t0, b - 10);
          document.getElementById('plotT0').value = a.toFixed(1);
          document.getElementById('plotT1').value = b.toFixed(1);
          document.getElementById('plotWid').value = (b - a).toFixed(1);
        }
        drawOv();
        loadWindow();
      });
    }).catch(function (e) { setPlotStatus('读取失败：' + e.message); });
  }

  // 自动跟随：仿真在跑 + 绘图页 + 勾选时，定时把窗口贴到最新日志尾部
  var followT = null;
  function tickFollow() {
    if (!document.getElementById('plotFollow').checked) return;
    if (S.tab !== 'plot') return;
    if (OFFLINE || !S.sim || S.sim.status !== 'running') return;
    GET('/api/plot/list').then(function (j) {
      var logs = j.logs || [];
      if (!logs.length) return;
      var names = logs.map(function (l) { return l.name; });
      // 跟随：选中「最近一个有数据行」的文件（跳过刚停/正在启动、只有表头的空文件）
      pickLatestWithData(names).then(function (latest) {
        if (latest !== PLOT.name) { openPlot(latest); return; }
        GET('/api/plot/meta?name=' + encodeURIComponent(latest)).then(function (m) {
          if (!m || !m.n) return;
          var b = m.t1, a = Math.max(m.t0, b - 10);
          document.getElementById('plotT0').value = a.toFixed(1);
          document.getElementById('plotT1').value = b.toFixed(1);
          document.getElementById('plotWid').value = (b - a).toFixed(1);
          GET('/api/plot/overview?name=' + encodeURIComponent(latest)).then(function (ov) { if (ov && ov.zmin) { PLOT.ov = ov; drawOv(); } });
          loadWindow();
        });
      });
    }).catch(function () { });
  }
  function ensureFollowTimer() {
    if (followT) return;
    followT = setInterval(tickFollow, 2500);
  }

  function loadWindow() {
    if (!PLOT.name) return;
    var a = +document.getElementById('plotT0').value, b = +document.getElementById('plotT1').value;
    if (!(b - a > 0)) { setPlotStatus('窗口无效'); return; }
    setPlotStatus('读取窗口…');
    GET('/api/plot/window?name=' + encodeURIComponent(PLOT.name) + '&t0=' + a + '&t1=' + b + '&thin=700')
      .then(function (seg) {
        if (!seg || !seg.time) { setPlotStatus('窗口内没有数据'); return; }
        PLOT.seg = seg; renderGrid(seg); markDirty(false);
        var span = seg.time[seg.time.length - 1] - seg.time[0];
        setPlotStatus(seg.file + ' · 窗口 ' + seg.time[0].toFixed(1) + '–' + seg.time[seg.time.length - 1].toFixed(1) + 's'
          + ' · ' + seg.rows + ' 步 · 抽稀 ' + seg.stride + ' 步/点 · ' + seg.npanel + ' 图/' + seg.ncurve + ' 曲线'
          + (seg.done ? ' · 窗口内 ' + seg.done + ' 步重置' : '') + ' · 全程 ' + (PLOT.ov ? (PLOT.ov.t1 - PLOT.ov.t0) : span).toFixed(0) + 's'
          + (PLOT.meta && PLOT.meta.sample ? ' · ⚠ 样例数据（非本次实验）' : ''));
      }).catch(function (e) { setPlotStatus('读取窗口失败：' + e.message); });
  }

  function markDirty(on) { var b = document.getElementById('plotDraw'); b.classList.toggle('warn', on); }

  // ---- 纵轴刻度（与 plot02 同：6 等分，步长只取 1/2/2.5/5×10^k）----
  var STEPS = (function () { var a = []; for (var e = -2; e <= 6; e++) [1, 2, 2.5, 5].forEach(function (m) { var s = m * Math.pow(10, e); if (Math.abs(s * 100 - Math.round(s * 100)) < 1e-9) a.push(+s.toFixed(4)); }); return Array.from(new Set(a)).sort(function (x, y) { return x - y; }); })();
  function step6(sp) { for (var i = 0; i < STEPS.length; i++) if (6 * STEPS[i] >= sp - 1e-12) return STEPS[i]; return STEPS[STEPS.length - 1]; }
  function stepUp(s) { var i = STEPS.findIndex(function (x) { return Math.abs(x - s) < 1e-12; }); return STEPS[Math.min(i + 1, STEPS.length - 1)]; }
  function niceAxis(lo, hi, sym) {
    if (sym) { var s = step6(2 * Math.max(Math.abs(lo), Math.abs(hi))); return [-3 * s, 3 * s, s]; }
    var s2 = step6(hi - lo), L = Math.floor(lo / s2 + 1e-9) * s2;
    while (L + 6 * s2 < hi - 1e-12) { s2 = stepUp(s2); L = Math.floor(lo / s2 + 1e-9) * s2; }
    return [L, L + 6 * s2, s2];
  }
  function decimals(st) { for (var d = 0; d <= 2; d++) if (Math.abs(st * Math.pow(10, d) - Math.round(st * Math.pow(10, d))) < 1e-9) return d; return 2; }

  function drawPanel(cv, p, seg, W, H, scale, fs, light) {
    var dpr = scale || window.devicePixelRatio || 1; fs = fs || 1;
    W = W || cv.clientWidth; H = H || cv.clientHeight; if (!W || !H) return;
    cv.width = W * dpr; cv.height = H * dpr;
    var g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    var C = light ? { fg: '#111827', muted: '#64748b', border: '#e2e8f0', card: '#ffffff' } : themeColors();
    var dark = light ? false : isDarkTheme();
    var kindSt = function (k, dflt) { var b = KIND[k] || dflt || KIND.in; return dark ? Object.assign({}, b, { color: KIND_DARK[k] || b.color }) : b; };
    var M = { l: Math.round(48 * fs), r: 8, t: Math.round(9 * fs), b: Math.round(16 * fs) }, iw = W - M.l - M.r, ih = H - M.t - M.b;
    var lo = Infinity, hi = -Infinity;
    p.series.forEach(function (s) { (seg.series[s.name] || []).forEach(function (v) { if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }); });
    var cmdLines = (p.cmd_line || []).map(function (cl) {
      var V = seg.series[cl.name]; if (!V) return null;
      if (!cl.over) { V.forEach(function (v) { if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }); return { st: kindSt(cl.kind), segs: null, vals: V }; }
      var O = seg.series[cl.over], segs = [];
      if (O) for (var i = 0, n = O.length; i < n;) { if (!O[i]) { i++; continue; } var j = i; while (j < n && O[j]) j++; var v = V[i]; segs.push([i, j, v]); if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } i = j; }
      return { st: kindSt(cl.kind), segs: segs };
    }).filter(Boolean);
    g.font = Math.round(10 * fs) + 'px system-ui';
    if (!isFinite(lo)) { g.fillStyle = C.muted; g.fillText('无该列数据', M.l, M.t + Math.round(12 * fs)); return; }
    var dec, ticks = null;
    if (p.ticks) { lo = p.ticks[0]; hi = p.ticks[p.ticks.length - 1]; ticks = p.ticks; dec = decimals(ticks.length > 1 ? (hi - lo) / (ticks.length - 1) : 1); }
    else {
      if (p.yrange) { lo = Math.min(p.yrange[0], lo); hi = Math.max(p.yrange[1], hi); }
      if (p.limit && Math.max(Math.abs(lo), Math.abs(hi)) > 0.5 * p.limit) { lo = Math.min(lo, -p.limit); hi = Math.max(hi, p.limit); }
      if (hi - lo < 1e-9) hi = lo + 1e-6;
      var ax = niceAxis(lo, hi, p.sym !== false); lo = ax[0]; hi = ax[1]; dec = decimals(ax[2]);
    }
    var last = seg.time.length - 1 || 1;
    var X = function (i) { return M.l + iw * i / last; }, Y = function (v) { return M.t + ih * (1 - (v - lo) / (hi - lo)); };

    (p.bands || []).forEach(function (bd) {
      var b = seg.series[bd.name]; if (!b) return;
      var bh = ih * (bd.h == null ? 1 : bd.h), by = M.t + ih - bh, n = b.length;
      var bcol = dark ? (BAND_DARK[bd.color] || bd.color) : bd.color;
      for (var i = 0; i < n;) {
        if (b[i] !== bd.value) { i++; continue; }
        var j = i; while (j < n && b[j] === bd.value) j++;
        var x0 = X(i), w = Math.max(X(Math.min(j, n - 1)) - x0, 0.6);
        g.fillStyle = bcol; g.fillRect(x0, by, w, bh);
        if (bh >= ih * 0.98) { g.font = Math.round(9 * fs) + 'px system-ui'; g.textAlign = 'left'; if (w - 6 >= g.measureText(bd.cn).width) { g.fillStyle = C.muted; g.fillText(bd.cn, x0 + 3, M.t + Math.round(9 * fs)); } }
        i = j;
      }
    });

    g.lineWidth = 1; g.fillStyle = C.muted;
    var gl = ticks || Array.from({ length: 7 }, function (_, k) { return lo + (hi - lo) * k / 6; });
    gl.forEach(function (v) {
      var y = Y(v); g.strokeStyle = C.border; g.beginPath(); g.moveTo(M.l, y); g.lineTo(W - M.r, y); g.stroke();
      g.textAlign = 'right'; g.fillText(fmt(v, dec), M.l - 5, y + Math.round(3 * fs));
    });
    g.textAlign = 'center'; g.fillStyle = C.muted;
    var nx = Math.max(2, Math.min(6, Math.floor(iw / (56 * fs))));
    for (var k = 0; k <= nx; k++) { var i = Math.round(last * k / nx); var txt = seg.time[i].toFixed(1), half = g.measureText(txt).width / 2; g.fillText(txt, Math.max(M.l + half, Math.min(X(i), W - M.r - half)), H - Math.round(4 * fs)); }
    g.textAlign = 'right'; g.fillStyle = C.muted; g.fillText(p.unit || '', W - M.r - 2, M.t + Math.round(9 * fs));

    if (p.limit) { g.strokeStyle = C.border; g.setLineDash([4, 4]); [p.limit, -p.limit].forEach(function (L) { if (L < lo || L > hi) return; g.beginPath(); g.moveTo(M.l, Y(L)); g.lineTo(W - M.r, Y(L)); g.stroke(); }); g.setLineDash([]); }

    cmdLines.forEach(function (cl) {
      var st = cl.st; g.save(); g.globalAlpha = st.a; g.strokeStyle = st.color; g.lineWidth = st.w; g.setLineDash(st.dash || []);
      if (cl.segs) { cl.segs.forEach(function (q) { var y = Y(q[2]); if (!(y >= M.t - 1 && y <= M.t + ih + 1)) return; var x0 = X(q[0]), x1 = Math.max(X(Math.min(q[1], seg.time.length - 1)), x0 + 1); g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke(); }); }
      else { g.beginPath(); cl.vals.forEach(function (v, j) { var x = X(j), y = Y(v); if (!j) g.moveTo(x, y); else { g.lineTo(x, Y(cl.vals[j - 1])); g.lineTo(x, y); } }); g.stroke(); }
      g.restore();
    });

    p.series.slice().sort(function (a, b) { return (KIND[a.kind] || { w: 0 }).w - (KIND[b.kind] || { w: 0 }).w; }).forEach(function (s) {
      var d = seg.series[s.name]; if (!d) return; var st = kindSt(s.kind, KIND.act);
      g.save(); g.globalAlpha = st.a; g.strokeStyle = st.color; g.lineWidth = st.w; g.setLineDash(st.dash || []); g.lineJoin = 'round'; g.beginPath();
      d.forEach(function (v, j) { var x = X(j), y = Y(v); if (!j) g.moveTo(x, y); else if (p.step) { g.lineTo(x, Y(d[j - 1])); g.lineTo(x, y); } else g.lineTo(x, y); });
      g.stroke(); g.restore();
    });
  }

  // ---- 相位图标题：鼠标扫过时浮动显示「编号↔中文」对照（纵轴本身只印数字）----
  var _phaseTip = null;
  function _ensurePhaseTip() {
    if (_phaseTip) return _phaseTip;
    _phaseTip = el('div', { class: 'phase-tip' });
    _phaseTip.style.display = 'none';
    document.body.appendChild(_phaseTip);
    return _phaseTip;
  }
  function attachPhaseTip(titleEl, legend) {
    titleEl.classList.add('has-tip');
    titleEl.title = '';   // 关掉原生 title，用自定义浮动窗
    var pinned = false;
    function show() {
      var tip = _ensurePhaseTip();
      tip.innerHTML = '';
      tip.appendChild(el('div', { class: 'phase-tip-h', text: '相位编号对照' }));
      var g = el('div', { class: 'phase-tip-grid' });
      legend.forEach(function (it) {
        var row = el('div', { class: 'phase-tip-row' });
        row.appendChild(el('b', { text: String(it.n) }));
        row.appendChild(el('span', { text: it.cn || '' }));
        g.appendChild(row);
      });
      tip.appendChild(g);
      tip.style.display = 'block';
      var r = titleEl.getBoundingClientRect();
      var tw = tip.offsetWidth, th = tip.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
      var x = r.left; if (x + tw > vw - 8) x = Math.max(8, vw - tw - 8);
      // 放在标题【上方】（占用上方「跳跃」区域，不遮挡下面的相位图像）；上方放不下才退回下方。
      var y = r.top - th - 6;
      if (y < 8) { y = r.bottom + 6; if (y + th > vh - 8) y = Math.max(8, vh - th - 8); }
      tip.style.left = x + 'px'; tip.style.top = y + 'px';
    }
    function hide() { if (pinned) return; if (_phaseTip) _phaseTip.style.display = 'none'; }
    titleEl.addEventListener('mouseenter', show);
    titleEl.addEventListener('mouseleave', hide);
    // 触屏无 hover：点标题切换固定/收起
    titleEl.addEventListener('click', function () {
      pinned = !pinned;
      if (pinned) { show(); } else if (_phaseTip) { _phaseTip.style.display = 'none'; }
    });
    // 点其它地方收起（点标题本身时上面已切换，这里用包含判断避免打架）
    document.addEventListener('click', function (e) {
      if (pinned && !titleEl.contains(e.target)) { pinned = false; if (_phaseTip) _phaseTip.style.display = 'none'; }
    });
    window.addEventListener('scroll', function () { if (!pinned && _phaseTip) _phaseTip.style.display = 'none'; }, true);
    window.addEventListener('resize', function () { if (!pinned && _phaseTip) _phaseTip.style.display = 'none'; });
  }

  function renderGrid(seg) {
    var grid = document.getElementById('plotGrid');
    grid.innerHTML = ''; PLOT.reg = [];
    var dark = isDarkTheme();
    var kcol = function (k) { var b = KIND[k] || KIND.act; return dark ? (KIND_DARK[k] || b.color) : b.color; };
    var ncol = seg.columns.length;
    grid.style.gridTemplateColumns = 'repeat(' + ncol + ', minmax(0,1fr))';
    seg.columns.forEach(function (col) {
      var box = el('div', { class: 'plotcolg' });
      col.forEach(function (p) {
        var c = el('div', { class: 'plotcell' });
        var h5 = el('h5', {}, [document.createTextNode(p.title), el('em', { text: p.unit || '' })]);
        if (p.phase_legend && p.phase_legend.length) attachPhaseTip(h5, p.phase_legend);
        c.appendChild(h5);
        var lg = el('div', { class: 'plotlg' + (p.lg_cls ? ' ' + p.lg_cls : '') });
        var _lg = {};
        p.series.slice().sort(function (a, b) { return (KIND[b.kind] || { w: 0 }).w - (KIND[a.kind] || { w: 0 }).w; }).forEach(function (s) {
          var st = KIND[s.kind] || KIND.act, sp = el('span', {});
          var col = kcol(s.kind);
          var bg = (st.dash && st.dash.length) ? 'repeating-linear-gradient(90deg,' + col + ' 0 3px,transparent 3px 6px)' : col;
          sp.appendChild(el('i', { style: 'background:' + bg + ';height:' + st.w + 'px;opacity:' + st.a }));
          sp.appendChild(document.createTextNode(s.cn)); _lg['s:' + s.name] = sp;
        });
        (p.bands || []).forEach(function (bd) { if (bd.legend === false) return; var sp = el('span', {}); var bc = dark ? (BAND_DARK[bd.color] || bd.color) : bd.color; sp.appendChild(el('i', { style: 'background:' + bc + ';height:9px;border:1px solid ' + (bd.border || (dark ? '#3a4757' : '#c9d3de')) })); sp.appendChild(document.createTextNode(bd.cn)); _lg['b:' + bd.name] = sp; });
        (p.cmd_line || []).forEach(function (cl) { var st = KIND[cl.kind] || KIND.in, col = kcol(cl.kind), sp = el('span', {}); sp.appendChild(el('i', { style: 'background:repeating-linear-gradient(90deg,' + col + ' 0 3px,transparent 3px 6px);height:' + st.w + 'px' })); sp.appendChild(document.createTextNode(cl.cn)); _lg['c:' + cl.name] = sp; });
        if (p.lg_rows) { p.lg_rows.forEach(function (row, i) { row.forEach(function (k) { if (_lg[k]) lg.appendChild(_lg[k]); }); if (i < p.lg_rows.length - 1) lg.appendChild(el('span', { style: 'flex-basis:100%;height:0' })); }); }
        else { Object.keys(_lg).forEach(function (k) { lg.appendChild(_lg[k]); }); }
        if (p.limit) lg.appendChild(el('span', { class: 'wn', textContent: '扭矩上限 ±' + p.limit.toFixed(1) }));
        c.appendChild(lg);
        var cv = el('canvas', {}); c.appendChild(cv); box.appendChild(c);
        c._w = p.weight || 1;
        PLOT.reg.push(function () { drawPanel(cv, p, seg); });
      });
      grid.appendChild(box);
    });
    // 各列总高对齐：同列按 weight 分摊
    var cols = Array.prototype.slice.call(grid.querySelectorAll('.plotcolg'));
    if (cols.length) {
      var H = Math.max.apply(null, cols.map(function (c) { return c.getBoundingClientRect().height; }));
      cols.forEach(function (col) {
        var cells = Array.prototype.slice.call(col.children), W = cells.reduce(function (s, c) { return s + c._w; }, 0);
        var gap = parseFloat(getComputedStyle(col).rowGap) || 8, avail = H - gap * (cells.length - 1);
        cells.forEach(function (c) { var cv = c.querySelector('canvas'); var want = avail * c._w / W - (c.offsetHeight - cv.clientHeight); if (want > 60) cv.style.height = want.toFixed(1) + 'px'; });
      });
    }
    PLOT.reg.forEach(function (f) { f(); });
  }

  // ---- 缩略轴：全程 body_z 包络 + 可拖窗口 ----
  function drawOv() {
    var cv = document.getElementById('plotOv'), dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (!W) return; cv.width = W * dpr; cv.height = H * dpr;
    var g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    var C = themeColors();
    if (!PLOT.ov || !PLOT.ov.zmin) return;
    var M = { l: 8, r: 8, t: 6, b: 12 }, iw = W - M.l - M.r, ih = H - M.t - M.b;
    var t0 = PLOT.ov.t0, t1 = PLOT.ov.t1 || t0 + 1, T = function (v) { return M.l + iw * (v - t0) / (t1 - t0); };
    var zlo = Math.min.apply(null, PLOT.ov.zmin), zhi = Math.max.apply(null, PLOT.ov.zmax); if (zhi - zlo < 1e-9) zhi = zlo + 1;
    var Z = function (v) { return M.t + ih * (1 - (v - zlo) / (zhi - zlo)); };
    g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath();
    PLOT.ov.t.forEach(function (t, i) { var x = T(t); i ? g.lineTo(x, Z(PLOT.ov.zmax[i])) : g.moveTo(x, Z(PLOT.ov.zmax[i])); });
    for (var i = PLOT.ov.t.length - 1; i >= 0; i--) g.lineTo(T(PLOT.ov.t[i]), Z(PLOT.ov.zmin[i]));
    g.closePath(); g.fillStyle = 'rgba(37,99,235,.16)'; g.fill();
    g.fillStyle = '#dc2626'; PLOT.ov.done.forEach(function (d, i) { if (d > 0) g.fillRect(T(PLOT.ov.t[i]) - 0.6, M.t, 1.6, ih); });
    var a = T(+document.getElementById('plotT0').value), b = T(+document.getElementById('plotT1').value);
    g.fillStyle = 'rgba(37,99,235,.14)'; g.fillRect(a, M.t, Math.max(b - a, 1), ih);
    g.strokeStyle = '#2563eb'; g.beginPath(); g.moveTo(a, M.t); g.lineTo(a, M.t + ih); g.moveTo(b, M.t); g.lineTo(b, M.t + ih); g.stroke();
    g.fillStyle = C.muted; g.font = '10px system-ui'; g.textAlign = 'left';
    g.fillText('body_z 全程包络 · 红＝重置 · 蓝框＝当前窗口（拖动/拉边界改窗口）', M.l + 2, M.t + 9);
  }
  function ovTime(ev) { var cv = document.getElementById('plotOv'), r = cv.getBoundingClientRect(); return PLOT.ov.t0 + (ev.clientX - r.left - 8) / (r.width - 16) * (PLOT.ov.t1 - PLOT.ov.t0); }
  function bindOv() {
    var cv = document.getElementById('plotOv');
    cv.addEventListener('mousedown', function (e) {
      if (!PLOT.ov) return;
      var t = ovTime(e), a = +document.getElementById('plotT0').value, b = +document.getElementById('plotT1').value, w = b - a, edge = 0.02 * (PLOT.ov.t1 - PLOT.ov.t0);
      PLOT.drag = Math.abs(t - a) < edge ? 'l' : Math.abs(t - b) < edge ? 'r' : (t > a && t < b) ? 'm' : ['n', t];
      if (PLOT.drag[0] === 'n') { setWin(t, t + w); PLOT.drag = ['m', t]; }
    });
    addEventListener('mousemove', function (e) {
      if (!PLOT.drag || !PLOT.ov) return; var t = ovTime(e);
      if (PLOT.drag === 'm') { var w = +document.getElementById('plotT1').value - +document.getElementById('plotT0').value; setWin(t - w / 2, t + w / 2); }
      else if (PLOT.drag === 'l') setWin(Math.min(t, +document.getElementById('plotT1').value - 0.2), +document.getElementById('plotT1').value);
      else if (PLOT.drag === 'r') setWin(+document.getElementById('plotT0').value, Math.max(t, +document.getElementById('plotT0').value + 0.2));
      drawOv();
    });
    addEventListener('mouseup', function () { PLOT.drag = null; });
  }
  function setWin(a, b) {
    var lo = PLOT.ov ? PLOT.ov.t0 : 0, hi = PLOT.ov ? PLOT.ov.t1 : 1e9;
    a = Math.max(lo, a); b = Math.min(hi, b);
    document.getElementById('plotT0').value = a.toFixed(1);
    document.getElementById('plotT1').value = b.toFixed(1);
    document.getElementById('plotWid').value = (b - a).toFixed(1);
    markDirty(true); drawOv();
  }

  // ---- 模型配置胶囊 ----
  function updateCfgChip() {
    var cfg = (PLOT.meta && PLOT.meta.config) || [];
    var nameEl = document.getElementById('plotCfgName');
    if (!cfg.length) { nameEl.textContent = '—'; }
    else if (cfg.length === 1) { nameEl.textContent = cfg[0].ckpt ? (cfg[0].model + ' / ' + cfg[0].ckpt) : cfg[0].model; }
    else { nameEl.textContent = cfg.map(function (c) { return c.model; }).join(' / '); }
    var chip = document.getElementById('plotCfgChip');
    chip.onclick = function (ev) {
      ev.stopPropagation();
      var exist = document.getElementById('cfgPop'); if (exist) { exist.remove(); return; }
      if (!cfg.length) { toast('这份录像没有记录模型配置', 'warn'); return; }
      var box = el('div', { class: 'cfgpop', id: 'cfgPop' });
      box.appendChild(el('h5', { text: '这份录像的模型配置' }));
      cfg.forEach(function (c) {
        box.appendChild(el('div', { class: 'line' }, [
          el('span', { class: 'st', text: c.role }),
          el('span', { class: 'nm', text: c.ckpt ? (c.model + ' / ' + c.ckpt) : c.model })
        ]));
      });
      var r = chip.getBoundingClientRect();
      box.style.position = 'fixed'; box.style.left = Math.max(8, r.left) + 'px'; box.style.top = (r.bottom + 6) + 'px';
      document.body.appendChild(box);
      setTimeout(function () { document.addEventListener('click', function close() { box.remove(); document.removeEventListener('click', close); }, { once: true }); }, 0);
    };
  }

  // ---- 导出（复用 plot02 的版式：固定尺寸单元格 + 表头）----
  var EXP_W = 660, EXP_H = 360, EXP_SCALE = 2, EXP_FS = 2, EXP_TS = 1.5;
  function headH(fs) { return Math.round(20 * EXP_TS) + Math.round(18 * fs); }
  function drawHeader(g, p, W, fs, ts) {
    var l1 = Math.round(20 * ts);
    g.fillStyle = '#111827'; g.font = '600 ' + Math.round(15 * ts) + 'px system-ui'; g.textAlign = 'center';
    g.fillText(p.title, W / 2, l1);
    var y2 = l1 + Math.round(13 * fs), x = W - Math.round(12 * fs);
    g.font = Math.round(12 * fs) + 'px system-ui';
    var items = p.series.slice();
    (p.cmd_line || []).forEach(function (cl) { items.push({ cn: cl.cn, kind: cl.kind || 'in' }); });
    items.sort(function (a, b) { return (KIND[b.kind] || { w: 0 }).w - (KIND[a.kind] || { w: 0 }).w; }).forEach(function (s) {
      var st = KIND[s.kind] || KIND.act;
      g.fillStyle = '#374151'; g.textAlign = 'right'; g.fillText(s.cn, x, y2);
      x -= g.measureText(s.cn).width + Math.round(5 * fs);
      g.fillStyle = st.color; g.fillRect(x - Math.round(18 * fs), y2 - Math.round(5 * fs), Math.round(18 * fs), Math.max(3, Math.round(3 * fs)));
      x -= Math.round(23 * fs);
    });
  }
  function exportCell(reg, W, H) {
    W = W || EXP_W; H = H || EXP_H; var hh = headH(EXP_FS);
    var out = el('canvas', {}); out.width = W * EXP_SCALE; out.height = (H + hh) * EXP_SCALE;
    var g = out.getContext('2d'); g.setTransform(EXP_SCALE, 0, 0, EXP_SCALE, 0, 0);
    g.fillStyle = '#fff'; g.fillRect(0, 0, W, H + hh); drawHeader(g, reg.p, W, EXP_FS, EXP_TS);
    var plot = el('canvas', {}); drawPanel(plot, reg.p, PLOT.seg, W, H, EXP_SCALE, EXP_FS, true);
    g.drawImage(plot, 0, hh, W, H); return out;
  }
  function stamp() { return (PLOT.seg ? PLOT.seg.file.replace(/\.tsv$/, '') : 'wbr') + '_' + (PLOT.seg ? PLOT.seg.time[0].toFixed(1) + '-' + PLOT.seg.time[PLOT.seg.time.length - 1].toFixed(1) : '') + 's'; }
  function compositeCanvas() {
    var cells = PLOT.reg, NC = PLOT.seg.columns.length, NR = Math.max.apply(null, PLOT.seg.columns.map(function (c) { return c.length; }));
    var cw = EXP_W * EXP_SCALE, ch = (EXP_H + headH(EXP_FS)) * EXP_SCALE, gap = 30, bar = 60;
    var out = el('canvas', {}); out.width = NC * cw + (NC + 1) * gap; out.height = bar + NR * ch + (NR + 1) * gap;
    var g = out.getContext('2d'); g.setTransform(1, 0, 0, 1, 0, 0);
    g.fillStyle = '#fff'; g.fillRect(0, 0, out.width, out.height);
    g.fillStyle = '#111827'; g.font = '600 28px system-ui'; g.textAlign = 'left';
    g.fillText('WBR 遥测全景  ' + stamp(), gap, bar - 20);
    var idx = 0;
    PLOT.seg.columns.forEach(function (col, ci) { col.forEach(function (p, ri) { g.drawImage(exportCell(cells[idx]), gap + ci * (cw + gap), bar + gap + ri * (ch + gap)); idx++; }); });
    return out;
  }
  function download(canvas, name) { return new Promise(function (res) { canvas.toBlob(function (b) { var a = el('a', { href: URL.createObjectURL(b), download: name }); document.body.appendChild(a); a.click(); a.remove(); setTimeout(function () { URL.revokeObjectURL(a.href); res(); }, 300); }, 'image/png'); }); }
  async function saveComposite() {
    if (!PLOT.reg.length) { toast('还没有可导出的图', 'warn'); return; }
    try { await download(compositeCanvas(), 'WBR遥测全景_' + stamp() + '.png'); toast('整页已下载', 'ok'); }
    catch (err) { toast('导出失败：' + err.message, 'bad'); }
  }
  async function saveAll() {
    if (!PLOT.reg.length) { toast('还没有可导出的图', 'warn'); return; }
    var seen = {}, cells = [];
    for (var i = 0; i < PLOT.reg.length; i++) {
      var base = (PLOT.reg[i].p.title || ('panel' + i)).replace(/[^\p{L}\p{N} _.-]/gu, '') + '_' + stamp();
      if (!seen[base]) seen[base] = 1; else { seen[base]++; base += '_' + seen[base]; }
      cells.push({ name: base + '.png', cv: exportCell(PLOT.reg[i]) });
    }
    setPlotStatus('正在导出 ' + cells.length + ' 张…');
    try {
      var r = await fetch('/api/plot/save', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dir: 'WBR遥测_' + stamp(), files: cells.map(function (c) { return { name: c.name, b64: c.cv.toDataURL('image/png').split(',')[1] }; }) })
      });
      var d = await r.json().catch(function () { return {}; });
      if (!r.ok || d.error) throw new Error(d.error || ('HTTP ' + r.status));
      toast('已存 ' + d.n + ' 张到 ' + d.dir, 'ok');
      setPlotStatus('已导出 ' + d.n + ' 张分图 → ' + d.dir);
    } catch (err) {
      toast('无本地后端，改为逐个下载…', 'warn');
      for (var j = 0; j < cells.length; j++) await download(cells[j].cv, cells[j].name);
      setPlotStatus('已触发 ' + cells.length + ' 次下载');
    }
  }

  /* ------------------------------------------------------------- 场景动作 */
  function loadScenario(sc) {
    ensureCheckpoints().then(function () {
      if (sc.slots) {   // uni 与老 switch 同构：有 slots 即按四阶段加载（老 switch 场景一并兼容）
        SLOT_ORDER.forEach(function (k) {
          var p = sc.slots[k]; if (!p) return;
          if (isScriptSlot(p)) { S.slots[k] = scriptSlotObj(); return; }   // 起立槽＝相位脚本（无权重）
          var c = S.checkpoints.find(function (x) { return x.path === p; });
          S.slots[k] = c ? { path: c.path, model: c.model, ckpt: c.ckpt, obs: c.obs, act: c.act }
            : { path: p, model: p.split('/').slice(-2, -1)[0], ckpt: p.split('/').pop(), obs: null, act: null };
        });
        setMode('uni');
      } else if (sc.single) {
        S.single = Object.assign({}, S.single, sc.single);
        setMode('single'); renderSingle();
      }
      S.scenarios.forEach(function (x) { x.active = (x.id === sc.id); });
      if (sc.params) { S.params = Object.assign({}, PARAM_DEFAULTS, S.params, sc.params); }
      document.getElementById('runScenarioPill').textContent = '场景：' + sc.name + ' ▾';
      renderChain(); renderChainParams(); persistSlots(); runPreflight(true);
      toast('已加载场景：' + sc.name, 'ok');
    });
  }

  function persistSlots() {
    var flat = {};
    SLOT_ORDER.forEach(function (k) { if (S.slots[k]) flat[k] = S.slots[k]; });
    POST('/api/settings', { slots: flat, single: S.single, mode: S.mode }).catch(function () { });
  }

  function openSaveScenario(sc) {
    var def = (sc && sc.name) ? sc.name : ('场景_' + nowStr().replace(/[: ]/g, '-'));
    S.saveCtx = sc ? { kind: 'scenario', forName: sc.name } : { kind: 'scenario' };
    document.getElementById('saveDlgTitle').textContent = sc ? ('覆写场景：' + sc.name) : '保存场景';
    document.getElementById('saveDlgHint').innerHTML = '保存到 <span class="mono">wbr_bench_store/scenarios/&lt;名称&gt;.json</span>';
    document.getElementById('saveName').value = def;
    var preview = buildSpec();
    preview.name = def;
    document.getElementById('savePreview').textContent = JSON.stringify(preview, null, 2);
    document.getElementById('dlgSaveScenario').showModal();
  }

  // 单策略区「固化检查点」：命名后把当前单策略 checkpoint 固化到检查点库
  function openSolidifyNamed() {
    var sp = S.single;
    if (!sp || !sp.path) { toast('单策略还没选 checkpoint', 'warn'); return; }
    if (!/\.pt$/.test(sp.path)) { toast('只能固化 .pt 权重（该 checkpoint 可能来自检查点库）', 'warn'); return; }
    var base = (sp.model || String(sp.path).split('/').slice(-2, -1)[0] || 'model')
      + '__' + String(sp.ckpt || String(sp.path).split('/').pop()).replace(/\.pt$/, '');
    S.saveCtx = { kind: 'saved' };
    document.getElementById('saveDlgTitle').textContent = '固化检查点';
    document.getElementById('saveDlgHint').innerHTML = '保存到 <span class="mono">wbr_bench_store/saved/&lt;名称&gt;/</span>（复制权重 + config.json 留档）';
    document.getElementById('saveName').value = base;
    document.getElementById('savePreview').textContent = JSON.stringify({
      name: base, source: sp.path, obs: sp.obs, act: sp.act, task: sp.task
    }, null, 2);
    document.getElementById('dlgSaveScenario').showModal();
  }

  // 检查点库条目「笔」：覆写到本条（用当前单策略 checkpoint 重新固化，沿用旧目录名）
  function openSolidifyOverwrite(s) {
    var sp = S.single;
    if (!sp || !sp.path || !/\.pt$/.test(sp.path)) { toast('先在右侧单策略区选好要固化的 checkpoint', 'warn'); return; }
    S.saveCtx = { kind: 'saved', forName: s.name };
    document.getElementById('saveDlgTitle').textContent = '覆写检查点：' + s.name;
    document.getElementById('saveDlgHint').innerHTML = '覆写到 <span class="mono">wbr_bench_store/saved/' + escapeHtml(s.name) + '/</span>（用当前单策略 checkpoint 重新固化）';
    document.getElementById('saveName').value = s.name;
    document.getElementById('savePreview').textContent = JSON.stringify({
      name: s.name, source: sp.path, obs: sp.obs, act: sp.act, task: sp.task
    }, null, 2);
    document.getElementById('dlgSaveScenario').showModal();
  }

  // 保存弹窗「保存」：按上下文分派（场景：新建/覆写；固化：新建/覆写）
  function submitSaveDialog() {
    var ctx = S.saveCtx || { kind: 'scenario' };
    var name = document.getElementById('saveName').value.trim() || ('场景_' + nowStr().replace(/[: ]/g, '-'));
    var dlg = document.getElementById('dlgSaveScenario');
    if (ctx.kind === 'saved') {
      var sp = S.single;
      if (!sp || !sp.path) { toast('单策略还没选 checkpoint', 'warn'); return; }
      POST('/api/saved', {
        path: sp.path, obs: sp.obs, act: sp.act, name: (ctx.forName || name)
      }).then(function (j) {
        dlg.close(); toast((ctx.forName ? '已覆写：' : '已固化：') + j.meta.name, 'ok'); loadSaved();
      }).catch(function (e) { toast('固化失败：' + e.message, 'bad'); });
      return;
    }
    var payload = buildSpec();
    payload.name = name;
    POST('/api/scenarios', { name: name, payload: payload }).then(function () {
      dlg.close(); toast((ctx.forName ? '已覆写场景：' : '场景已保存：') + name, 'ok');
      loadScenarios().then(function () { document.getElementById('runScenarioPill').textContent = '场景：' + name + ' ▾'; });
    }).catch(function (e) { toast(e.message, 'bad'); });
  }

  // 删除确认弹窗
  function confirmDelete(kind, name, label, extra) {
    S.delCtx = { kind: kind, name: name };
    document.getElementById('delDlgTitle').textContent = '确认删除';
    document.getElementById('delDlgMsg').innerHTML = '确定删除' + label + ' <b>' + escapeHtml(name) + '</b> 吗？'
      + (extra || '') + '<br><span class="muted" style="font-size:.72rem">此操作不可撤销。</span>';
    document.getElementById('dlgConfirmDel').showModal();
  }
  function submitDeleteDialog() {
    var c = S.delCtx; if (!c) return;
    var dlg = document.getElementById('dlgConfirmDel');
    var done = function (ok, msg) {
      dlg.close(); toast(msg || (ok ? '已删除' : '删除失败'), ok ? 'ok' : 'bad');
      if (c.kind === 'scenario') loadScenarios(); else loadSaved();
    };
    if (c.kind === 'scenario') {
      DEL('/api/scenarios', { name: c.name }).then(function () { done(true, '已删除场景'); }).catch(function (e) { done(false, e.message); });
    } else {
      DEL('/api/saved', { name: c.name }).then(function () { done(true, '已删除存档'); }).catch(function (e) { done(false, e.message); });
    }
  }
  function confirmDeleteScenario(sc) {
    confirmDelete('scenario', sc.name, '场景', (sc.mode === 'single' ? '（单策略）' : '（双模式）'));
  }
  function confirmDeleteSaved(s) {
    confirmDelete('saved', s.name, '固化存档', '（源：' + escapeHtml(s.source || '?') + '）');
  }

  // 检查点库条目双击：加载到单策略
  function loadSavedToSingle(s) {
    if (!s.source) { toast('该存档没有记录源路径，无法加载', 'warn'); return; }
    ensureCheckpoints().then(function () {
      var dir = String(s.source).split('/').slice(0, -1).join('/');
      S.single = {
        task: taskForObs(s.obs, s.source),
        run: dir,
        checkpoint: parseInt((String(s.ckpt || s.source.split('/').pop()).match(/(\d+)/) || [0, 0])[1], 10),
        from_run_config: dir,
        path: s.source,
        obs: s.obs,
        act: s.act
      };
      setMode('single'); renderSingle();
      runPreflight(true);
      toast('单策略 ← ' + s.name, 'ok');
    });
  }

  function escapeHtml(t) {
    return String(t == null ? '' : t).replace(/[&<>"']/g, function (ch) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch];
    });
  }

  function setMode(mode) {
    S.mode = mode; persistSlots();
    applyModeToModelPage(); renderSingle();
  }

  /* --------------------------------------------------- 引擎选择（Sim2Sim P2） */
  function engineCN(e) { return e === 'mujoco' ? 'MuJoCo' : 'Isaac Gym'; }
  // 运行中以桥上报的实际引擎为准（分段控件此时锁定）；空闲显示用户的选择
  function displayEngine() {
    if (S.sim && S.sim.status === 'running' && S.state && S.state.engine) return S.state.engine;
    return S.engine;
  }
  function applyEngineUI() {
    var eng = displayEngine();
    var running = S.sim && S.sim.status === 'running';
    document.body.classList.toggle('engine-mujoco', eng === 'mujoco');
    $$('#engineSeg [data-engine]').forEach(function (b) {
      b.setAttribute('aria-pressed', String(b.getAttribute('data-engine') === eng));
      b.disabled = running;
    });
    renderEngineBadge();
  }
  function renderEngineBadge() {
    var b = document.getElementById('engineBadge'); if (!b) return;
    var running = S.sim && S.sim.status === 'running';
    if (!running) { b.textContent = '○ ENGINE: 未启动'; b.className = 'engine-badge'; return; }
    b.textContent = '● ENGINE: ' + engineCN(displayEngine());
    b.className = 'engine-badge lit ' + (displayEngine() === 'mujoco' ? 'is-mujoco' : 'is-isaac');
  }
  function mjPersist() {
    cfxRead();
    try { localStorage.setItem(MJ_CFG_KEY, JSON.stringify(S.mj)); } catch (e) { }
  }
  function buildMujocoSpec() {
    cfxRead();
    var m = S.mj;
    var slot = mjSlot();
    var cmds = { lin_vel_x: +m.vx || 0, ang_vel_yaw: +m.yaw || 0,
                 base_height: +m.h || slot.defH, ang_roll: +m.roll || 0,
                 jump_height: +m.jump || 0.10 };
    if (slot.chain) {                                        // 五槽链：契约/资产/权重全在链里
      return { engine: 'mujoco', mode: 'mujoco', wmode: 'uni', start: slot.start,
               terrain: !!m.terrain,
               cmds: { lin_vel_x: +m.vx || 0, base_height: slot.defH,
                       ang_roll: +m.roll || 0, jump_height: +m.jump || 0.10 },
               dur: +m.dur || 0, record: !!m.record, window: !!m.window, speed: +m.speed || 1 };
    }
    var spec = {
      engine: 'mujoco', mode: 'mujoco', wmode: mjSlotKey(),
      contract: slot.contract,                               // 契约透传：serve 按此加载
      cmds: cmds,
      dur: +m.dur || 0, record: !!m.record, window: !!m.window, speed: +m.speed || 1
    };
    if (m.customPt) { spec.model = m.customPt; }
    else { spec.release = m.release || null; spec.model = m.model || null; }
    if (m.mjcf) spec.mjcf = m.mjcf;
    return spec;
  }

  /* ------------------------------------------------ MuJoCo 配置页签（并入主导航） */
  // 与运行页快启卡共用 S.mj / localStorage wbr.mj.cfg；输入用 cfx* id，互不踩。
  function cfxRead() {
    var g = function (id) { return document.getElementById(id); };
    if (!g('cfxVx')) return;
    S.mj.vx = parseFloat(g('cfxVx').value) || 0;
    S.mj.yaw = parseFloat(g('cfxYaw').value) || 0;
    S.mj.h = parseFloat(g('cfxH').value) || mjSlot().defH;
    S.mj.roll = parseFloat(g('cfxRoll').value) || 0;
    S.mj.dur = parseFloat(g('cfxDur').value) || 0;
    S.mj.record = g('cfxRecord').checked;
    S.mj.window = g('cfxWindow').checked;
    S.mj.speed = parseFloat(g('cfxSpeed').value) || 1;
    S.mj.preset = g('cfxPreset').value;
    S.mj.slot = (g('cfxSlot') && g('cfxSlot').value) || S.mj.slot;
    S.mj.jump = parseFloat((g('cfxJump') || {}).value) || S.mj.jump || 0.10;
    if (g('cfxTerrain')) S.mj.terrain = g('cfxTerrain').checked;
    S.mj.customPt = g('cfxCustomPt').value.trim();
    var sel = g('cfxRelease');
    S.mj.release = sel.value || '';
    var opt = sel.selectedIndex >= 0 ? sel.options[sel.selectedIndex] : null;
    S.mj.model = (opt && opt.getAttribute('data-model')) || '';   // release 下拉里选中的 ckpt 文件名
    S.mj.mjcf = g('cfxMjcf').value || '';
  }
  function cfxRestore() {
    var g = function (id) { return document.getElementById(id); };
    if (!g('cfxVx')) return;
    cfxRenderMode();
    cfxFillPresetOptions();
    g('cfxVx').value = S.mj.vx; g('cfxYaw').value = S.mj.yaw; g('cfxH').value = S.mj.h;
    g('cfxRoll').value = S.mj.roll || 0;
    g('cfxDur').value = S.mj.dur; g('cfxRecord').checked = !!S.mj.record; g('cfxWindow').checked = !!S.mj.window;
    g('cfxSpeed').value = String(S.mj.speed || 1);
    g('cfxCustomPt').value = S.mj.customPt || '';
    if (g('cfxTerrain')) g('cfxTerrain').checked = !!S.mj.terrain;
  }
  // —— 槽位切换（S3）：换契约、清选型、换 MJCF、量程/预设随槽位重建
  function mjSetSlot(key) {
    if (!MJ_SLOTS[key] || S.mj.slot === key) return;
    S.mj.slot = key;
    var sl = MJ_SLOTS[key];
    S.mj.release = ''; S.mj.model = '';
    S.mj.mjcf = sl.mjcf || '';
    S.mj.h = sl.defH;
    S.mj.roll = 0;
    S.mj.start = sl.start || '2w';
    if (!mjPresets()[S.mj.preset]) S.mj.preset = 'stand';
    cfxRestore();
    cfxFillPresetOptions();
    cfxFillModels();
    cfxFillManifest();
    cfxRenderMode();
    mjPersist();
  }
  function cfxRenderMode() {
    var sel = document.getElementById('cfxSlot');
    if (sel) sel.value = mjSlotKey();
    var sl = mjSlot();
    var box = document.getElementById('cfxChainBox');
    if (box) box.hidden = !sl.chain;
    var tbox = document.getElementById('cfxTerrainBox');
    if (tbox) tbox.hidden = !sl.chain;      // 地形只对五槽链开放（单槽各自契约）
    var cs = document.getElementById('cfxChainStart');
    if (cs) cs.value = sl.start || '2w';
    var rollEl = document.getElementById('cfxRoll');
    if (rollEl) {
      rollEl.disabled = !sl.roll;
      rollEl.title = sl.roll ? '两轮 roll 指令范围 [-0.2, 0.2]' : '四轮/趴下 roll 指令范围退化（置灰）';
    }
    var rel = document.getElementById('cfxRelease');
    if (rel) rel.disabled = !!sl.chain;
    var pt = document.getElementById('cfxCustomPt');
    if (pt) pt.disabled = !!sl.chain;
  }
  function cfxFillPresetOptions() {
    var sel = document.getElementById('cfxPreset'); if (!sel) return;
    var ps = mjPresets();
    sel.innerHTML = '';
    Object.keys(ps).forEach(function (k) {
      sel.appendChild(el('option', { value: k, text: ps[k].label || k }));
    });
    if (!ps[S.mj.preset]) S.mj.preset = 'stand';
    sel.value = S.mj.preset;
  }
  function cfxFillModels() {
    var sel = document.getElementById('cfxRelease'); if (!sel) return;
    var sl = mjSlot();
    var wantObs = sl.obs;
    sel.innerHTML = '';
    if (sl.chain) {
      sel.appendChild(el('option', { text: '（链内固定四槽权重，见下方契约栏）', value: '' }));
      S.mj.release = ''; S.mj.model = '';
      mjPersist();
      return;
    }
    sel.appendChild(el('option', { text: '（契约默认交付：' + (sl.release || '—') + '）', value: '' }));
    var groups = {};
    (S.checkpoints || []).forEach(function (c) {
      if (c.obs !== wantObs) return;
      var d = c.path.split('/').slice(0, -1).join('/');
      (groups[d] = groups[d] || []).push(c);
    });
    var dirs = Object.keys(groups).sort();
    dirs.forEach(function (d) {
      var best = groups[d].sort(function (a, b) { return (b.mtime || 0) - (a.mtime || 0); })[0];
      var o = el('option', { text: d.split('/').pop() + ' · ' + best.ckpt + ' · obs ' + best.obs + '/act ' + best.act, value: d });
      o.setAttribute('data-model', best.ckpt);
      sel.appendChild(o);
    });
    if (S.mj.release && groups[S.mj.release]) sel.value = S.mj.release;
    else { sel.value = ''; S.mj.release = ''; S.mj.model = ''; }
    mjPersist();
  }
  function cfxFillManifest() {
    if (!document.getElementById('cfxMjcf')) return Promise.resolve();
    return GET('/api/mujoco/manifest').then(function (m) {
      var sel = document.getElementById('cfxMjcf');
      sel.innerHTML = '';
      (m.mjcfs || []).forEach(function (f) {
        sel.appendChild(el('option', { text: f.name + (f.mesh ? '（网格）' : '（无网格）'), value: f.path }));
      });
      // 槽位自带资产优先：持久化里存的资产与当前槽不一致时（如旧的两轮 wbr_v1）纠正回槽位资产
      var want = mjSlot().mjcf;
      if (want) {
        var wantBase = want.split('/').pop();
        var curBase = (S.mj.mjcf || '').split('/').pop();
        if (curBase !== wantBase) {
          for (var i = 0; i < sel.options.length; i++) {
            if (sel.options[i].value.split('/').pop() === wantBase) { sel.selectedIndex = i; S.mj.mjcf = sel.value; break; }
          }
        }
      }
      if (S.mj.mjcf) sel.value = S.mj.mjcf; else if (sel.options.length) sel.selectedIndex = 0;
      document.getElementById('cfxMjcfSha').textContent = 'sha ' + (m.mjcfs && m.mjcfs[sel.selectedIndex] ? m.mjcfs[sel.selectedIndex].sha256 : '—');
      var ck = mjSlot().chain ? 'four' : mjSlotKey();
      var c = (m.contracts || {})[ck] || (m.contracts || {}).four || m.contract || {};
      var hrg = (c.command_ranges || {}).base_height || [];
      document.getElementById('cfxContract').innerHTML = '契约（' + mjSlot().cn + '）：obs <b>' + (c.obs_dim || '—') + '</b> / act <b>' + (c.act_dim || '—') +
        '</b> · 默认交付 <b>' + (c.release || '—') + '</b> / <b>' + (c.release_model || '—') +
        '</b> · Height 量程 <b>' + hrg.join('–') + '</b>';
      var hEl = document.getElementById('cfxH');               // 高度量程随契约收放
      if (hEl && hrg.length === 2) { hEl.min = hrg[0]; hEl.max = hrg[1]; }
      cfxRenderMode();
      var p = ((m.parities || {})[ck]) || m.parity || {};
      var worst = function (kind) {
        var sc = p.scenarios || {}; var v = 0;
        Object.keys(sc).forEach(function (k) { var x = sc[k][kind]; if (x && x.max > v) v = x.max; });
        return v;
      };
      var g = p.gates || {};
      document.getElementById('cfxParity').innerHTML =
        '<b>' + (p.passed ? '✅ PASS' : '❌ 未过') + '</b> · 场景 <b>' + Object.keys(p.scenarios || {}).length + '</b> 组' +
        ' · obs 最差 <b>' + worst('obs').toExponential(1) + '</b>（门 ' + (g.obs != null ? g.obs : '—') + '）' +
        ' · act 最差 <b>' + worst('act').toExponential(1) + '</b>（门 ' + (g.act != null ? g.act : '—') + '）' +
        ' · τ 最差 <b>' + worst('tau').toExponential(1) + '</b>（门 ' + (g.tau != null ? g.tau : '—') + '）' +
        (p.policy ? '<br>权重 <span class="mono">' + (p.policy.ckpt || '').split('/').slice(-2).join('/') + '</span> · iter <b>' + p.policy.iter + '</b> · sha <span class="mono">' + String(p.policy.sha256 || '').slice(0, 8) + '</span>' : '');
    }).catch(function () { document.getElementById('cfxParity').textContent = '对账数据读取失败（eval/golden/parity.json）'; });
  }
  function cfxRenderChecks(items) {
    var box = document.getElementById('cfxChecks'); if (!box) return;
    box.innerHTML = '';
    (items || []).forEach(function (p) {
      var mark = p.level === 'ok' ? '✓' : p.level === 'warn' ? '⚠' : p.level === 'bad' ? '✗' : '·';
      box.appendChild(el('span', { class: p.level, style: 'font-size:.74rem;display:block' },
        [el('b', { text: mark }), document.createTextNode(' ' + p.text)]));
    });
  }
  function cfxStart() {
    mjPersist();
    POST('/api/preflight', { spec: buildMujocoSpec() }).then(function (j) {
      cfxRenderChecks(j.items);
      var bad = (j.items || []).filter(function (i) { return i.level === 'bad'; }).length;
      var doLaunch = function () {
        POST('/api/start', { spec: buildMujocoSpec() }).then(function (r) {
          toast('已启动：' + r.run + '，切到运行页控制', 'ok');
          switchTab('run');
        }).catch(function (e) {
          var p = e.payload || {};
          if (p.items) cfxRenderChecks(p.items);
          toast('启动失败：' + (p.text || e.message), 'bad');
        });
      };
      if (bad > 0) {
        if (window.confirm('预检有 ' + bad + ' 项不通过，仍要强制启动吗？')) doLaunch();
        else toast('已取消（先解决预检项）', 'warn');
      } else doLaunch();
    }).catch(function (e) { toast('预检失败：' + e.message, 'bad'); });
  }

  // —— 置灰矩阵（设计稿 §3）：单点实现。要锁的控件在 app.html/renderNav 标 data-mj-locked
  //    （+data-mj-why 文案），视觉由 CSS body.engine-mujoco [data-mj-locked] 统一处理；
  //    这里只装两个守卫：capture 级委托点击（拦截动作、弹一次提示）＋ 键盘（M/H/U/O/J/L）。
  function mjGuardToast(msg) {
    if (applyEngineUI._t) return;
    toast(msg, 'warn');
    applyEngineUI._t = true;
    setTimeout(function () { applyEngineUI._t = false; }, 1800);
  }

  /* ------------------------------------------------------------------ 启动 */
  function doStart(force) {
    if (OFFLINE) { toast('离线预览不能启动仿真', 'warn'); return; }
    var spec = (S.engine === 'mujoco') ? buildMujocoSpec() : buildSpec();
    POST('/api/start', { spec: spec, force: !!force }).then(function (j) {
      toast('已启动：' + j.run, 'ok'); refreshStatus();
    }).catch(function (e) {
      var p = e.payload || {};
      if (p.error === 'training' || (p.items && p.items.some(function (i) { return i.level === 'bad'; }))) {
        if (p.items) renderPreflight(p.items);
        var why = p.text || '预检未通过';
        if (window.confirm(why + '\n\n仍要强制启动吗？')) { doStart(true); }
        else toast(why, 'bad');
      } else { toast('启动失败：' + e.message, 'bad'); }
    });
  }

  function doStop() {
    POST('/api/stop', {}).then(function () { toast('已停止', 'ok'); refreshStatus(); }).catch(function (e) { toast(e.message, 'bad'); });
  }

  var lastSimStatus = null;
  function refreshStatus() {
    if (OFFLINE) { renderSimState(); return; }
    GET('/api/status').then(function (j) {
      // 【Sim2Sim】仿真自己退出（如 MuJoCo 终止判定摔倒）不再无声：拉尾日志给一行原因
      if (lastSimStatus === 'running' && j.status === 'idle') {
        var rid = (S.sim && S.sim.run) || '';
        GET('/api/runlog?id=' + encodeURIComponent(rid)).then(function (r) {
          var lines = (r.text || '').split('\n').filter(function (l) { return l.trim(); });
          var tail = lines.slice(-2).join(' ｜ ');
          if (/终止判定|Traceback|Error/.test(tail)) toast('仿真已退出：' + tail.slice(-180), 'bad');
          else toast('仿真已退出', 'warn');
        }).catch(function () { toast('仿真已退出', 'warn'); });
      }
      lastSimStatus = j.status;
      S.sim = j; renderSimState(); renderStatus(S.state);
    }).catch(function () { });
  }

  var pollT = null;
  function pollState() {
    if (OFFLINE || S.sim.status !== 'running') return;
    GET('/api/state').then(function (st) {
      if (st.sim) { S.sim = st.sim; renderSimState(); }
      if (st.bridgeDown) return;
      S.state = st; updateReadouts(st);
      // 【A+B】每拍刷新指令回执条与动作按钮解禁状态（守卫表来自桥）
      renderStatus(st);
      if (S.tab === 'plot') updateCfgChip();
    }).catch(function () { });
  }

  /* --------------------------------------------------- 原生窗口停靠（P3） */
  var DOCK = { on: false, polling: false, timer: null, tries: 0, rect: null, cal: null, calTries: 0, calT: null, lastSeen: null };
  var dockWatch = null;

  // 【Sim2Sim】停靠保持看门狗：DOCK.on 期间每 3 s 拿窗口实测几何与目标比对，漂移 >8px 静默重贴
  //（MuJoCo viewer 初始化后期挪窗 / WM 干预 / 误拖后都能自动回位；Isaac 同样受益）。
  function dockWatchStart() {
    clearInterval(dockWatch);
    dockWatch = setInterval(function () {
      if (!DOCK.on || S.sim.status !== 'running' || !DOCK.rect) return;
      GET('/api/window').then(function (j) {
        var st = (j.status || {});
        if (!st.found) return;                    // 窗口暂不可查（终止/最小化）：不动，等仿真层处理
        var r = DOCK.rect;
        if (Math.abs(st.x - r.x) > 8 || Math.abs(st.y - r.y) > 8 ||
            Math.abs((st.w || 0) - r.w) > 8 || Math.abs((st.h || 0) - r.h) > 8) {
          dockWindow(false);
        }
      }).catch(function () { });
    }, 3000);
  }

  function setVpStatus(text) {
    var c = document.getElementById('vpCorner');
    if (c) c.textContent = text;
  }

  // 视口在屏幕上的矩形（设备像素）。返回 null 表示这个视图拿不到可信的屏幕坐标。
  // 嵌入式面板（OpenChamber 的浏览器面板/webview）里 window.screenX/outerWidth 读不到真值
  // （实测 screenX=0、outerWidth=0、outerHeight=0）。真浏览器窗口 outerWidth/outerHeight 都 >0。
  // 【旧判据 chromeX>60||chromeY>160 会误杀】Edge 垂直标签页侧栏让 outerWidth-innerWidth 轻松 >300，
  //   真浏览器标签页被当成嵌入面板 ⇒「标签页不能停、独立窗口能停」。现判据：
  //   ① outerWidth/outerHeight 都 ≥ 200（嵌入 webview 恒为 0，这是决定性特征）；② 视口矩形足够大。
  //   不再要求 screenX 非 0（最大化的真浏览器在 (0,0) 时 screenX/screenY 也是 0，不能据此否决）。
  //   是否「矩形下面真有窗口」交给后端 rect_inside_window 兜底（X11 实测）。
  function vpRectDevice() {
    var vp = document.getElementById('viewport');
    if (!vp) return null;
    var r = vp.getBoundingClientRect();
    if (r.width < 160 || r.height < 120) return null;
    // ① 已做 2 点校准：用校准反解出的 CSS→设备像素 线性变换。多显示器 + 分数缩放下最可靠，
    //    绕开 window.screenX/screenY 是「相对所在显示器」的 CSS 像素这个坑（第 2 块屏会整体偏移）。
    if (DOCK.xform && DOCK.xform.ok) {
      var X = DOCK.xform;
      return {
        x: Math.round(X.ox + X.s * r.left),
        y: Math.round(X.oy + X.s * r.top),
        w: Math.round(X.s * r.width),
        h: Math.round(X.s * r.height),
        xform: true
      };
    }
    // ② 回退：单显示器 / 1x 缩放用 window.screenX 估算（多屏/分数缩放不准，故建议先校准）
    if (!window.outerWidth || !window.outerHeight ||
        window.outerWidth < 200 || window.outerHeight < 200) return null;
    var chromeX = Math.max(0, (window.outerWidth || 0) - (window.innerWidth || 0));
    var chromeY = Math.max(0, (window.outerHeight || 0) - (window.innerHeight || 0));
    var dpr = window.devicePixelRatio || 1;
    // 窗口是否最大化/贴边：最大化时窗口外框基本铺满屏幕，窗口内起始点 screenX/screenY≈0，
    // 视口相对窗口内容区的偏移要按「内容区原点」算；非最大化时窗口居中，左右各有 chromeX/2 边框。
    var maxW = (window.screen && window.screen.availWidth) ? window.screen.availWidth : 0;
    var maxH = (window.screen && window.screen.availHeight) ? window.screen.availHeight : 0;
    var maxed = (maxW && window.outerWidth >= maxW - 4) || (maxH && window.outerHeight >= maxH - 4) ||
                (Math.abs(window.screenX) <= 2 && Math.abs(window.screenY) <= 2);
    var baseX = maxed ? window.screenX : (window.screenX + chromeX / 2);
    return {
      x: Math.round((baseX + r.left) * dpr),
      y: Math.round((window.screenY + chromeY + r.top) * dpr),
      w: Math.round(r.width * dpr),
      h: Math.round(r.height * dpr),
      maxed: maxed
    };
  }

  function dockWindow(manual) {
    if (OFFLINE) { if (manual) toast('离线预览不能停靠', 'warn'); return; }
    // 【P8 五槽】对齐的三级目标来源（vpRectDevice 原逻辑保留）：
    //   ① DOCK.xform（校准对齐 2 点解算，或点击校准自动解出——见 /api/point 应答）⇒ 精确；
    //   ② 浏览器 screenX 估算（多屏/分数缩放下不准）⇒ 尽力而为。
    var rect = vpRectDevice();
    if (!rect) {
      setVpStatus('仿真视口 · 该视图拿不到屏幕坐标');
      if (manual) toast('当前视图是嵌入式面板，拿不到屏幕坐标；用普通浏览器窗口打开本页或点「独立窗口」', 'warn');
      return;
    }
    var send = { x: Math.round(rect.x + (DOCK.cal ? DOCK.cal.x : 0)),
                 y: Math.round(rect.y + (DOCK.cal ? DOCK.cal.y : 0)),
                 w: rect.w, h: rect.h };
    setVpStatus('仿真视口 · 定位中 x' + send.x + ' y' + send.y + ' ' + send.w + '×' + send.h);
    POST('/api/window/dock', send).then(function (j) {
      var st = j.status || {};
      if (j.needcal) { DOCK.on = false; setVpStatus('仿真视口 · 校准不足——用鼠标点几处不同按钮后重试'); return; }
      if (!st.found) {
        DOCK.on = false; setVpStatus('仿真视口 · 未找到仿真窗口');
        if (manual) toast('没找到 Isaac Gym 窗口（可能还在启动）', 'warn');
        return;
      }
      DOCK.on = true; DOCK.rect = rect;
      dockWatchStart();
      // 自校正：比对窗口实际左上角与目标（二者都是设备像素），偏差 > 6px 就累计校准偏移重贴（最多 3 次，防抖）。
      // 【Sim2Sim】只认**小偏差**（<200px）为换算误差；大偏差是 WM 覆盖/窗口未就位（实测 MuJoCo 新窗
      // 会被 mutter 初始放置覆盖，一秒内又回目标位）——此时目标没错，不能纠偏，交给看门狗重贴拉回。
      if (st.x != null && st.y != null) {
        var errX = st.x - send.x;
        var errY = st.y - send.y;
        if ((Math.abs(errX) > 6 || Math.abs(errY) > 6) &&
            Math.abs(errX) < 200 && Math.abs(errY) < 200 && (DOCK.calTries || 0) < 3) {
          DOCK.cal = { x: (DOCK.cal ? DOCK.cal.x : 0) - errX, y: (DOCK.cal ? DOCK.cal.y : 0) - errY };
          DOCK.calTries = (DOCK.calTries || 0) + 1;
          clearTimeout(DOCK.calT);
          DOCK.calT = setTimeout(function () { dockWindow(false); }, 150);
          setVpStatus('仿真视口 · 对齐中（校准 ' + DOCK.calTries + '）');
          return;
        }
        DOCK.calTries = 0;
      }
      setVpStatus('仿真视口 · 已停靠（' + st.w + '×' + st.h + '）');
    }).catch(function (e) {
      var msg = (e.payload && e.payload.error) || e.message || '停靠失败';
      if (/可见窗口/.test(msg)) {
        // 后端说「矩形下面没有窗口」——把算出的坐标显示出来便于现场核对（真浏览器标签页也该到这一步）
        setVpStatus('仿真视口 · 坐标不可信 x' + send.x + ' y' + send.y + ' ' + send.w + '×' + send.h);
        if (manual) toast('停靠失败：算出的视口矩形 x' + send.x + ' y' + send.y + ' ' + send.w + '×' + send.h
          + ' 下面没有可见窗口。请确认用真实浏览器标签页打开（非嵌入面板），或按 Ctrl+Alt+Shift+D 打印本机窗口列表', 'bad');
      } else {
        setVpStatus('仿真视口 · 停靠失败');
        if (manual) toast('停靠失败：' + msg, 'bad');
      }
    });
  }

  function undockWindow(silent) {
    DOCK.on = false; DOCK.rect = null;
    clearInterval(DOCK.timer); DOCK.polling = false;
    clearInterval(dockWatch);
    if (OFFLINE) return;
    POST('/api/window/undock', {}).catch(function () { });
    if (!silent) setVpStatus('仿真视口 · 已降下');
  }

  // 窗口要等 Isaac Gym 建好才出现：轮询 /api/window 直到 found 再停靠。
  // 注意 renderSimState 每 600ms 会调一次，这里必须自己去重，否则轮询被反复清掉。
  function dockWhenReady(manual) {
    if (!manual && (DOCK.on || DOCK.polling)) return;
    DOCK.polling = true;
    clearInterval(DOCK.timer);
    DOCK.tries = 0;
    DOCK.lastSeen = null;
    DOCK.timer = setInterval(function () {
      if (S.tab !== 'run' || (S.sim.status !== 'running' && !manual)) {
        DOCK.polling = false; clearInterval(DOCK.timer); return;
      }
      if (DOCK.tries++ > 60) {
        DOCK.polling = false; clearInterval(DOCK.timer);
        if (manual) toast('等不到 Isaac Gym 窗口', 'warn');
        return;
      }
      GET('/api/window').then(function (j) {
        if (j.status && j.status.found) {
          var st = j.status;
          // 【Sim2Sim】贴前等几何稳定：连续两拍一致才贴（MuJoCo viewer 初始化会先建窗再挪，贴早了白贴）
          var g = [st.x, st.y, st.w, st.h, st.mapped].join(',');
          if (g === DOCK.lastSeen) {
            DOCK.polling = false; clearInterval(DOCK.timer); dockWindow(manual);
          } else {
            DOCK.lastSeen = g;
          }
        }
      }).catch(function () { });
    }, 1000);
  }

  // 视口尺寸变了（窗口缩放/导航模式切换）→ 重新贴合
  var dockResizeT = null;
  var lastVpSize = null;
  function onViewportResize() {
    if (!DOCK.on) return;
    var vp = document.getElementById('viewport');
    if (vp) {
      var w = Math.round(vp.getBoundingClientRect().width), h = Math.round(vp.getBoundingClientRect().height);
      if (lastVpSize && Math.abs(w - lastVpSize.w) < 8 && Math.abs(h - lastVpSize.h) < 8) return;
      lastVpSize = { w: w, h: h };
    }
    clearTimeout(dockResizeT);
    dockResizeT = setTimeout(function () { dockWindow(false); }, 600);
  }

  /* ---- 2 点指针校准（多显示器 / 分数缩放下的窗口对齐）----
     问题：window.screenX/screenY 是「相对所在显示器」的 CSS 像素，X11 停靠用的是全局设备像素；
     多屏（且各屏缩放不同，如 1.33× + 2×）时，视口矩形会整体偏移/缩放错位。
     解法：进入校准后，在「仿真视口区」内点两下（建议左上、右下）。每次用真实鼠标事件的
     clientX/clientY（与 getBoundingClientRect 同一坐标系，也含屏幕取整），配合后端 XQueryPointer
     给的全局设备像素坐标，反解线性变换 设备 = (ox,oy) + s·client（等比缩放 s）。
     之后停靠/实时跟随都用该变换定位，跨屏/分数缩放都能精确对齐。 */
  function cancelCal() {
    if (DOCK.calStep) {
      DOCK.calStep = null;
      var vp = document.getElementById('viewport'); if (vp) vp.classList.remove('calibrating');
      toast('已取消校准', 'warn');
    }
  }
  function startCalibrate() {
    if (OFFLINE) { toast('离线预览不能校准', 'warn'); return; }
    // 停靠时原生窗口盖住视口，无法在视口上点击 ⇒ 先降下窗口，露出可点击的视口区再校准
    if (DOCK.on) { undockWindow(true); }
    DOCK.calStep = { samples: [] };
    var vp = document.getElementById('viewport');
    if (vp) vp.classList.add('calibrating');
    setVpStatus('校准 1/2：在「仿真视口区」左上角点一下');
    toast('校准：在仿真视口区左上角点一下，再在右下角点一下（Esc 取消）', 'warn');
  }
  function calCapture(ev) {
    if (!DOCK.calStep) return;
    ev.preventDefault(); ev.stopPropagation();
    var vp = document.getElementById('viewport');
    var r = vp.getBoundingClientRect();
    // clientX/clientY 与 getBoundingClientRect 同坐标系（视口坐标，含屏幕取整），作 CSS 侧真值
    var cx = ev.clientX, cy = ev.clientY;
    GET('/api/window/pointer').then(function (j) {
      if (!j || !j.ok) { toast('读不到指针坐标（无 X 显示？）', 'bad'); cancelCal(); return; }
      DOCK.calStep.samples.push({ cx: cx, cy: cy, dx: j.x, dy: j.y });
      var n = DOCK.calStep.samples.length;
      if (n < 2) {
        setVpStatus('校准 2/2：在「仿真视口区」右下角再点一下');
        toast('再在视口区右下角点一下', 'warn');
        return;
      }
      var a = DOCK.calStep.samples[0], b = DOCK.calStep.samples[1];
      var dcx = b.cx - a.cx, dcy = b.cy - a.cy;
      var ddx = b.dx - a.dx, ddy = b.dy - a.dy;
      var okx = Math.abs(dcx) > 30, oky = Math.abs(dcy) > 30;
      if (!okx && !oky) { toast('两点太近，请重来（拉开到视口两角）', 'bad'); DOCK.calStep = { samples: [] }; return; }
      // 用像素跨度更大的轴估缩放比（抗单轴取整误差）
      var s = (Math.abs(dcx) >= Math.abs(dcy)) ? (ddx / dcx) : (ddy / dcy);
      if (!(s > 0.2 && s < 8)) { toast('校准异常（s=' + s.toFixed(2) + '），请重来', 'bad'); DOCK.calStep = { samples: [] }; return; }
      var ox = a.dx - s * a.cx;
      var oy = a.dy - s * a.cy;
      DOCK.calStep = null;
      if (vp) vp.classList.remove('calibrating');
      DOCK.xform = { ok: true, s: s, ox: ox, oy: oy };
      DOCK.cal = { x: 0, y: 0 }; DOCK.calTries = 0;
      toast('校准完成（s=' + s.toFixed(3) + '），正在对齐…', 'ok');
      setVpStatus('校准完成 s=' + s.toFixed(3));
      dockWhenReady(true);
    }).catch(function (e) { toast('校准失败：' + (e.message || e), 'bad'); cancelCal(); });
  }

  // 【P8 五槽】浏览器最小化/遮挡 → 停靠的 Isaac 窗口自动降下（置顶无边框窗会悬空卡死，
  //   无法移动/最小化——11-04 作者实测）；恢复可见 → 自动重新贴合。
  document.addEventListener('visibilitychange', function () {
    if (OFFLINE) return;
    if (document.hidden) {
      if (DOCK.on) undockWindow(true);
    } else if (S.tab === 'run' && S.sim && S.sim.status === 'running') {
      DOCK.on = false;
      dockWhenReady(false);
    }
  });

  // 浏览器窗口被拖动时没有事件：每 2s 比对一次矩形，变了就重新贴合
  var dockAssert = 0;
  setInterval(function () {
    if (!DOCK.on || S.tab !== 'run') { dockAssert = 0; return; }
    dockAssert += 1;
    var r = vpRectDevice();
    if (!r || !DOCK.rect) return;
    var moved = Math.abs(r.x - DOCK.rect.x) > 4 || Math.abs(r.y - DOCK.rect.y) > 4 ||
      Math.abs(r.w - DOCK.rect.w) > 4 || Math.abs(r.h - DOCK.rect.h) > 4;
    // 【P8 五槽】矩形变了立即重贴；每 ~10s 无条件重贴一次（防 WM 拉回/瞬态失败后不自愈）
    if (moved || dockAssert >= 5) { dockAssert = 0; dockWindow(false); }
  }, 2000);


  // 统一动作入口（按钮与快捷键共用）
  function doAction(a) {
    if (a === 'armjump') {
      // 桥/链守卫是唯一事实源：链模式下由 allowed 表决定，单槽由 canJump 决定。离线预览拦截。
      if (OFFLINE) { toast('离线预览', 'warn'); return; }
      POST('/api/action', { action: a }).then(function (r) {
        if (r.msg && r.msg !== a) toast(r.msg, 'warn'); else toast('已下发：' + a, 'ok');
        pollState();
      }).catch(function (e) { toast(e.message, 'bad'); });
      return;
    }
    if (OFFLINE) { toast('离线预览', 'warn'); return; }
    POST('/api/action', { action: a }).then(function (r) {
      if (r.msg && r.msg !== a) toast(r.msg, 'warn'); else toast('已下发：' + a, 'ok');
      pollState();
    }).catch(function (e) { toast(e.message, 'bad'); });
  }

  // 键盘改通道值：按步长增减，夹在量程内，下发并刷新读数
  function nudgeChannel(key, delta) {
    var spec = CHANNELS.find(function (c) { return c.key === key; }); if (!spec) return;
    var modeKey = (S.state && S.state.modeKey) || null;
    if (spec.key === 'jump_height' && isSlotMode()) {
      if (modeKey !== 'TWOW') { toast('仅两轮模式可调跳跃', 'warn'); return; }
    } else if (spec.switchOff && isSlotMode()) { toast('双模式未用该通道', 'warn'); return; }
    var e = chanEls[key]; if (!e) return;
    var v = parseFloat(e.inp.value) + delta;
    v = Math.min(spec.max, Math.max(spec.min, v));
    v = Math.round(v / spec.step) * spec.step;
    e.inp.value = v; e.vTxt.textContent = fmt(v, spec.d);
    e.tri.children[0].textContent = 'in ' + fmt(v, spec.d);
    if (key === 'ang_vel_yaw' && yawTemp.base !== null) yawTemp.base = v;   // 临时偏航期间 a/d 改值 ⇒ 记为新基值
    if (OFFLINE) return;
    POST('/api/cmd', { key: key, value: v }).catch(function () { });
  }

  // 指令重置到「当前模式默认位」：滑块与下发值一起回默认（四轮 h=0.19、两轮 h=0.28，其余归 0）
  // 当前「默认高度」的唯一出处：初始姿态 / 滑条默认显示 / 空格复位三处共用
  function currentDefH() {
    var st = S.state || {};
    var mj = (st.engine === 'mujoco') || (S.engine === 'mujoco');
    if (mj) {
      if (st.chain) return (st.slot === 'IDLE2W') ? 0.25 : 0.19;   // 链：两轮槽 0.25，四轮/趴下 0.19
      var sl = mjSlot(); return (sl && sl.defH) || 0.19;
    }
    var mk = st.modeKey || 'FOURW';
    return (mk === 'TWOW') ? 0.28 : 0.19;
  }
  function resetChannelsToDefault() {
    var defH = currentDefH();
    yawTemp.held = 0; yawTemp.base = null;   // 清临时偏航态（重置后的 yaw=0 即为设定值）
    var home = { lin_vel_x: 0, ang_vel_yaw: 0, base_height: defH, ang_roll: 0 };
    Object.keys(home).forEach(function (key) {
      var spec = CHANNELS.find(function (c) { return c.key === key; }); if (!spec) return;
      var e = chanEls[key]; if (!e) return;
      var v = Math.min(spec.max, Math.max(spec.min, home[key]));
      e.inp.value = v; e.vTxt.textContent = fmt(v, spec.d);
      e.tri.children[0].textContent = 'in ' + fmt(v, spec.d);
      if (!OFFLINE) POST('/api/cmd', { key: key, value: v }).catch(function () { });
    });
    if (!OFFLINE) toast('指令已重置到默认位（h=' + defH.toFixed(3) + '）', 'ok');
  }

  // 运行页快捷键：W/S A/D Q/E(按住临时偏航±0.5) J/L 通道；U/O 跳跃高度 −/+；H 装填跳；空格 指令回默认；M 切模式；R 复位
  function handleRunKey(e) {
    if (S.tab !== 'run') return false;
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    // 焦点在按钮上时，空格留给按钮默认行为（避免抢掉按钮激活）
    if (e.key === ' ' && e.target && e.target.closest && e.target.closest('button')) return false;
    var k = e.key, handled = true;
    switch (k) {
      case 'w': case 'W': nudgeChannel('lin_vel_x', 0.05); break;
      case 's': case 'S': nudgeChannel('lin_vel_x', -0.05); break;
      case 'a': case 'A': nudgeChannel('ang_vel_yaw', -0.05); break;
      case 'd': case 'D': nudgeChannel('ang_vel_yaw', 0.05); break;
      case 'q': case 'Q': yawTempStart(-1, 1); break;   // 按住临时偏航 −0.5，松开恢复
      case 'e': case 'E': yawTempStart(1, 2); break;    // 按住临时偏航 +0.5，松开恢复
      case 'i': case 'I': nudgeChannel('base_height', 0.005); break;
      case 'k': case 'K': nudgeChannel('base_height', -0.005); break;
      case 'j': case 'J': nudgeChannel('ang_roll', -0.02); break;
      case 'l': case 'L': nudgeChannel('ang_roll', 0.02); break;
      case 'u': case 'U': nudgeChannel('jump_height', -0.01); break;
      case 'o': case 'O': nudgeChannel('jump_height', 0.01); break;
      case 'h': case 'H': doAction('armjump'); break;
      case ' ': resetChannelsToDefault(); break;
      case 'm': case 'M': doAction('toggle'); break;
      case 'r': case 'R': doAction('reset'); break;
      default: handled = false;
    }
    return handled;
  }

  function bind() {
    // 【Sim2Sim P3】置灰矩阵守卫：capture 级拦截 + 键盘（置灰不隐藏，动作不执行只提示）
    document.addEventListener('click', function (ev) {
      if (!document.body.classList.contains('engine-mujoco')) return;
      var n = ev.target && ev.target.closest ? ev.target.closest('[data-mj-locked]') : null;
      if (!n) return;
      ev.preventDefault(); ev.stopPropagation();
      mjGuardToast(n.getAttribute('data-mj-why') || 'MuJoCo 下不可用');
    }, true);
    document.addEventListener('keydown', function (ev) {
      if (!document.body.classList.contains('engine-mujoco')) return;
      var k = (ev.key || '').toLowerCase();
      if (['m', 'h', 'u', 'o', 'j', 'l'].indexOf(k) < 0) return;   // 模式/装填/跳跃高度/roll
      // 【A+B】链模式：快捷键全部放行（动作键 → doAction，通道键 → nudgeChannel）
      if (S.state && S.state.chain) return;
      if (k === 'h' && S.state && S.state.canJump) return;          // 跳跃槽允许装填
      var tag = ((ev.target && ev.target.tagName) || '').toLowerCase();
      if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
      mjGuardToast('MuJoCo 单策略（四轮）不支持该操作');
    }, true);
    // 【Sim2Sim P2】引擎分段控件：只切启动面板；运行中被拒
    $$('#engineSeg [data-engine]').forEach(function (b) {
      b.addEventListener('click', function () {
        var eng = b.getAttribute('data-engine');
        if (eng === S.engine) return;
        if (S.sim && S.sim.status === 'running') { toast('先停止仿真，再切换引擎', 'warn'); return; }
        S.engine = eng; store.set('engine', eng);
        applyEngineUI();
        runPreflight(true);
      });
    });
    // 【Sim2Sim】MuJoCo 配置页签（cfx*）：输入持久化 + 预检/导出/启动
    var mjBind = function (id, ev) {
      var n = document.getElementById(id); if (n) n.addEventListener(ev, mjPersist);
    };
    ['cfxVx', 'cfxYaw', 'cfxH', 'cfxRoll', 'cfxJump', 'cfxDur', 'cfxCustomPt', 'cfxSpeed', 'cfxRelease', 'cfxMjcf', 'cfxTerrain'].forEach(function (id) { mjBind(id, 'change'); });
    mjBind('cfxRecord', 'change'); mjBind('cfxWindow', 'change');
    var slotSel = document.getElementById('cfxSlot');
    if (slotSel) slotSel.addEventListener('change', function () { mjSetSlot(slotSel.value); });
    var csSel = document.getElementById('cfxChainStart');
    if (csSel) csSel.addEventListener('change', function () {
      var k = S.mj.slot === 'uni4' ? 'uni4' : 'uni';
      S.mj.start = csSel.value;
      MJ_SLOTS[k].start = csSel.value;
      mjPersist();
    });
    var cfxP = document.getElementById('cfxPreset');
    if (cfxP) cfxP.addEventListener('change', function () {
      var p = mjPresets()[cfxP.value]; if (!p) return;
      document.getElementById('cfxVx').value = p.vx;
      document.getElementById('cfxYaw').value = p.yaw;
      document.getElementById('cfxH').value = p.h;
      mjPersist();
    });
    var cfxBtn = document.getElementById('cfxCheckBtn');
    if (cfxBtn) cfxBtn.addEventListener('click', function () {
      mjPersist();
      POST('/api/preflight', { spec: buildMujocoSpec() }).then(function (j) {
        cfxRenderChecks(j.items);
        var bad = (j.items || []).filter(function (i) { return i.level === 'bad'; }).length;
        toast(bad ? ('预检有 ' + bad + ' 项不通过') : '预检通过', bad ? 'bad' : 'ok');
      }).catch(function (e) { toast('预检失败：' + e.message, 'bad'); });
    });
    var cfxExp = document.getElementById('cfxExportBtn');
    if (cfxExp) cfxExp.addEventListener('click', function () {
      cfxExp.disabled = true;
      POST('/api/mujoco/export', {}).then(function (j) {
        var w = window.open('', '_blank');
        if (w) w.document.write('<title>' + j.path + '</title><body style="margin:0;background:#111;display:flex;align-items:center;justify-content:center"><img style="max-width:100%;max-height:100%" src="' + j.dataUrl + '"></body>');
        else toast('已导出 ' + j.path, 'ok');
      }).catch(function (e) { toast('导出失败：' + (e.payload && e.payload.error || e.message), 'bad'); })
        .finally(function () { cfxExp.disabled = false; });
    });
    var cfxStartBtn = document.getElementById('cfxStartBtn');
    if (cfxStartBtn) cfxStartBtn.addEventListener('click', cfxStart);
    // 导航位置/配色的绑定在 renderTopCtrls 里随块重建（块会在顶部/右侧间搬迁），此处不再重复绑定
    $$('[data-vp]').forEach(function (b) {
      b.addEventListener('click', function () {
        var k = b.getAttribute('data-vp');
        if (k === 'fullscreen') {
          if (document.fullscreenElement) document.exitFullscreen();
          else document.documentElement.requestFullscreen && document.documentElement.requestFullscreen();
        } else if (k === 'camera') {
          if (OFFLINE) { toast('离线预览不能控制相机', 'warn'); return; }
          var want = !(S.state && S.state.camera);
          POST('/api/action', { action: 'camera', value: want }).then(function () {
            toast('相机跟随已' + (want ? '开启' : '关闭'), 'ok');
            pollState();
          }).catch(function (e) { toast('相机开关失败：' + ((e.payload && e.payload.error) || e.message), 'bad'); });
        } else if (k === 'front' || k === 'realign') {
          dockWhenReady(true);
        } else if (k === 'calib') {
          startCalibrate();
        } else if (k === 'popout') {
          // 嵌入式面板里 window.open 仍然拿不到屏幕坐标 ⇒ 让后端起一个真实 --app 窗口
          if (OFFLINE) { window.open(location.origin + '/?tab=run&dock=1', 'wbr_console', 'width=1280,height=900'); return; }
          POST('/api/app/open', { url: location.origin + '/?tab=run&dock=1' }).then(function () {
            toast('已在新窗口打开控制台，那里会自动停靠', 'ok');
          }).catch(function (e) {
            toast('起不了独立窗口：' + ((e.payload && e.payload.error) || e.message), 'bad');
          });
        } else if (k === 'undock') {
          undockWindow();
        }
      });
    });
    $$('[data-act]').forEach(function (b) {
      b.addEventListener('click', function () { doAction(b.getAttribute('data-act')); });
    });
    document.getElementById('tglAuto').addEventListener('change', function (e) {
      if (OFFLINE) return;
      POST('/api/action', { action: 'auto', value: e.target.checked }).catch(function () { });
      toast(e.target.checked ? '连跳：开' : '连跳：关', 'ok');
    });
    document.getElementById('tglLoop') && document.getElementById('tglLoop').addEventListener('change', function (e) { toast(e.target.checked ? '循环：开（P3）' : '循环：关', 'ok'); });
    document.getElementById('btnFavorite').addEventListener('click', favoriteCurrent);
    document.getElementById('btnCheck').addEventListener('click', function () { runPreflight(false); });
    document.getElementById('btnCheck2').addEventListener('click', function () { runPreflight(false); });
    document.getElementById('btnStart').addEventListener('click', function () { doStart(false); });
    document.getElementById('btnStart2').addEventListener('click', function () { doStart(false); });
    document.getElementById('btnStop').addEventListener('click', doStop);
    document.getElementById('runScenarioPill').addEventListener('click', function () { switchTab('model'); toast('在「模型配置 → 场景管理」选择场景', 'ok'); });
    $$('.cfg-card').forEach(function (c) { c.addEventListener('click', function () { setMode(c.getAttribute('data-mode')); }); });
    $$('.cfg-use').forEach(function (b) { b.addEventListener('click', function (e) { e.stopPropagation(); setMode(b.getAttribute('data-mode')); }); });
    var smSel = document.getElementById('standupModeSel');
    if (smSel) smSel.addEventListener('change', function () {
      S.params.standup_mode = smSel.value; persistParams(); renderChainParams(); runPreflight(true);
    });
    document.getElementById('btnSaveScenario').addEventListener('click', function () { openSaveScenario(null); });
    document.getElementById('saveConfirm').addEventListener('click', submitSaveDialog);
    document.getElementById('delConfirm').addEventListener('click', submitDeleteDialog);
    document.getElementById('btnSinglePick').addEventListener('click', function () {
      ensureCheckpoints().then(function () { S.pickerSlot = 'single'; openPicker('single'); });
    });
    document.getElementById('btnSingleSolidify').addEventListener('click', openSolidifyNamed);
    ['pickerSearch', 'pickerSource', 'pickerFavOnly', 'pickerMatchOnly'].forEach(function (id) {
      document.getElementById(id).addEventListener('input', renderPicker);
      document.getElementById(id).addEventListener('change', renderPicker);
    });
    document.getElementById('singleTask').addEventListener('change', function (e) { S.single.task = e.target.value; persistSlots(); renderSingle(); runPreflight(true); });
    document.getElementById('singleRobots').addEventListener('change', function (e) { S.single.robots = parseInt(e.target.value, 10) || 3; persistSlots(); runPreflight(true); });
    document.getElementById('singleCkpt').addEventListener('change', function (e) {
      var v = e.target.value;
      S.single.checkpoint = (v === '' ? null : parseInt(v, 10));
      // 保持 path 与 checkpoint 一致：若已知 run 目录，则重建 ckpt 路径
      if (S.single.run && S.single.checkpoint != null) S.single.path = S.single.run + '/model_' + S.single.checkpoint + '.pt';
      persistSlots(); renderSingle(); runPreflight(true);
    });
    $$('[data-close]').forEach(function (b) { b.addEventListener('click', function () { b.closest('dialog').close(); }); });
    document.getElementById('btnHelp') && document.getElementById('btnHelp').addEventListener('click', function () { document.getElementById('dlgHelp').showModal(); });
    document.addEventListener('keydown', function (e) {
      if (e.target.matches('input, select, textarea, [contenteditable]')) return;
      // 停靠诊断：Ctrl+Alt+Shift+D 打印本机可见窗口列表 + 本页几何（真浏览器里核对停靠用）
      if (e.ctrlKey && e.altKey && e.shiftKey && (e.key === 'D' || e.key === 'd')) {
        e.preventDefault();
        var g = 'screenX=' + window.screenX + ' screenY=' + window.screenY
          + ' outer=' + window.outerWidth + '×' + window.outerHeight
          + ' inner=' + window.innerWidth + '×' + window.innerHeight
          + ' dpr=' + window.devicePixelRatio;
        var rect = vpRectDevice();
        console.log('[dock diag]', g, 'rect=', rect);
        toast('页面几何：' + g + (rect ? '；视口矩形 x' + rect.x + ' y' + rect.y + ' ' + rect.w + '×' + rect.h : '；视口矩形不可得'), 'warn');
        GET('/api/window/list').then(function (j) {
          var ws = (j && j.windows) || [];
          console.log('[dock diag] visible windows:', ws);
          toast('可见窗口 ' + ws.length + ' 个：' + ws.slice(0, 6).map(function (w) {
            return (w.isaac ? '★' : '') + (w.title || '无题') + '@' + w.x + ',' + w.y + ' ' + w.w + '×' + w.h;
          }).join(' | '), 'warn');
        }).catch(function () { });
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      // 数字键 1..N 按导航页签顺序切换（与 TABS 视觉顺序一致，改顺序不用改这里）
      if (/^[1-9]$/.test(e.key)) {
        var ti = parseInt(e.key, 10) - 1;
        if (ti < TABS.length) { switchTab(TABS[ti].key); return; }
      }
      if (handleRunKey(e)) { e.preventDefault(); }
    });
    // 【P7F4】Q/E 临时偏航的松开恢复（与 keydown 在同一开关块，焦点过滤一致）
    document.addEventListener('keyup', function (e) {
      if (e.target && e.target.matches && e.target.matches('input, select, textarea, [contenteditable]')) return;
      if (e.key === 'q' || e.key === 'Q') yawTempEnd(1);
      else if (e.key === 'e' || e.key === 'E') yawTempEnd(2);
    });
    // 窗口缩放/导航模式切换后重新贴合原生窗口
    window.addEventListener('resize', onViewportResize);
    if (window.ResizeObserver) { try { new ResizeObserver(onViewportResize).observe(document.getElementById('viewport')); } catch (e) { } }
    // 校准模式：在视口区点击两下捕获指针真实设备坐标（见 startCalibrate）
    var vpCal = document.getElementById('viewport');
    if (vpCal) vpCal.addEventListener('click', function (ev) { if (DOCK.calStep) calCapture(ev); }, true);
    document.addEventListener('keydown', function (ev) { if (ev.key === 'Escape' && DOCK.calStep) cancelCal(); });
  }

  /* -------------------------------------------------------------- 数据加载 */
  function loadScenarios() { return GET('/api/scenarios').then(function (j) { S.scenarios = j.scenarios || []; renderScenarios(); return S.scenarios; }).catch(function () { return []; }); }
  function loadSaved() { return GET('/api/saved').then(function (j) { S.saved = j.saved || []; renderSaved(); }).catch(function () { }); }

  function bootstrapFromSettings(settings) {
    S.settings = settings || {};
    S.slots = {};
    SLOT_ORDER.forEach(function (k) {
      var s = (settings.slots || {})[k];
      S.slots[k] = s ? (s.script ? scriptSlotObj()
        : { path: s.path, model: s.model, ckpt: s.ckpt, obs: s.obs, act: s.act })
        : { path: null, model: '（未选择）', ckpt: '—', obs: null, act: null };
    });
    if (settings.single) S.single = Object.assign(S.single, settings.single);
    if (settings.params) S.params = Object.assign({}, PARAM_DEFAULTS, settings.params);
    if (settings.mode && settings.mode !== 'switch') S.mode = settings.mode;   // 只保留 uni / single，旧的 switch 归一为默认 uni
  }

  function defaultsFromMock() {
    // 离线预览：用 mock 里的四槽
    S.slots = {};
    (MOCK.slots || []).forEach(function (s) { S.slots[s.key] = { path: s.path, model: s.model, ckpt: s.ckpt, obs: s.obs, act: s.act }; });
    S.scenarios = (MOCK.scenarios || []).slice();
    S.saved = (MOCK.saved || []).slice();
  }

  function boot() {
    applyTheme(S.theme); bindThemeWatch();
    renderNav(); setNavPos(S.navpos); switchTab(S.tab);
    renderChannels(); renderSimState(); renderStatus(null); renderDrawer();

    GET('/api/status').then(function (st) {
      S.sim = st;
      return Promise.all([GET('/api/settings'), loadScenarios(), loadSaved()]);
    }).then(function (res) {
      bootstrapFromSettings(res[0]);
      // 没存过四槽/单策略时，用第一个场景当默认（避免首屏四个“未选择”）
      if (!Object.keys(S.settings.slots || {}).length && S.scenarios.length) {
        var def = S.scenarios.find(function (x) { return x.mode === 'uni'; }) || S.scenarios.find(function (x) { return x.slots; }) || S.scenarios[0];
        if (def && def.slots) {
          SLOT_ORDER.forEach(function (k) {
            var p = def.slots[k]; if (!p) return;
            if (isScriptSlot(p)) { S.slots[k] = scriptSlotObj(); return; }
            S.slots[k] = { path: p, model: p.split('/').slice(-2, -1)[0], ckpt: p.split('/').pop(), obs: null, act: null };
          });
        }
      }
      var active = S.scenarios.find(function (x) { return x.active; }) || S.scenarios[0];
      if (active) {
        // 刷新后活动场景要真正套用（不只是换 pill 文字）：slots + params 与「加载」同逻辑，
        // 否则 pill 说 P7T/P7S 而参数却是 settings 旧值，启动结果与场景名不符（2026-10-26 修复）。
        if (active.slots) {
          SLOT_ORDER.forEach(function (k) {
            var p = active.slots[k]; if (!p) return;
            if (isScriptSlot(p)) { S.slots[k] = scriptSlotObj(); return; }   // 起立槽＝相位脚本（无权重）
            S.slots[k] = { path: p, model: p.split('/').slice(-2, -1)[0], ckpt: p.split('/').pop(), obs: null, act: null };
          });
        }
        if (active.params) { S.params = Object.assign({}, PARAM_DEFAULTS, S.params, active.params); }
        document.getElementById('runScenarioPill').textContent = '场景：' + active.name + ' ▾';
      }
      renderChain(); renderChainParams(); renderSingle(); applyModeToModelPage();
      applyEngineUI();
      renderSimState(); runPreflight(true);
      ensureCheckpoints().then(function () { enrichSlots(); renderChain(); renderSingle(); runPreflight(true); });
      document.getElementById('brandSub').textContent = 'WBR 控制台 · 在线';
      var rbs = document.getElementById('railBrandSub'); if (rbs) rbs.textContent = 'WBR 控制台 · 在线';
      pollT = setInterval(function () { refreshStatus(); pollState(); }, 600);
    }).catch(function () {
      // 后端不可达 ⇒ 离线预览
      OFFLINE = true;
      if (MOCK) defaultsFromMock();
      renderChain(); renderChainParams(); renderSingle(); applyModeToModelPage();
      renderScenarios(); renderSaved(); renderSimState(); renderStatus(null);
      document.getElementById('brandSub').textContent = 'WBR 控制台 · 离线预览';
      var rbs2 = document.getElementById('railBrandSub'); if (rbs2) rbs2.textContent = 'WBR 控制台 · 离线预览';
      toast('后端不可达：当前为离线预览（python wbr_bench.py 可启动）', 'warn');
    });

    bind();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
