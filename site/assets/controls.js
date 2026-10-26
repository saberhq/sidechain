/* controls.js: the control-cells page (content/vcc2026-control-cells.md, layouts/controls.html). Three stacked
   sections drawn with plotly.js from the files scripts/site_controls.py writes under static/data/controls/: the maps
   (or the QC violins of the same cells), the guides against the shuffles, the highest-expressed genes. The rail's
   cards pick a context for all three. The colour plan comes from the palette in index.json, the chrome from the
   site's tokens. Every visit opens on the same view (all six contexts, UMAP, by cluster); the chips are not kept
   between visits. The lower sections draw when they near the screen. */
(() => {
  'use strict';
  const app = document.getElementById('cc-app');
  if (!app) return;
  const BASE = app.dataset.base, PLOTLY = app.dataset.plotly, SRI = app.dataset.plotlySri;
  /* cells: val | test | all (one joint map) or a letter (that context's own map). color: what the map is coloured by;
     measure and by: the violins' measure and grouping. A colour chip moves one of the two, so a switch of plot keeps
     the last thing picked. */
  const DEF = { cells: 'all', plot: 'umap', resid: 'off', color: 'cluster', measure: 'total_umi', by: 'context',
    g_ctx: 'A', gn_ctx: 'A', fold_cl: '', fold_g: '' };
  const S = Object.assign({}, DEF);
  let IDX = null, PAL = {};

  /* ── small helpers ── */
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fin = v => v != null && isFinite(v);
  const fmt = (v, d) => !fin(v) ? '–' : (Math.abs(v) >= 1000 || (d === undefined && Number.isInteger(+v))) ? Math.round(v).toLocaleString('en-US') :
    (+v).toFixed(d ?? (Math.abs(v) < 1 ? 3 : Math.abs(v) < 10 ? 2 : 1));
  const trim = s => s.replace(/\.0+(?=k?$)/, '');
  const short = v => !fin(v) ? '–' : Math.abs(v) >= 1000 ? trim((v / 1000).toFixed(v >= 1e4 ? 0 : 1)) + 'k' : Math.abs(v) >= 1 ? trim(fmt(v, 1)) : String(+v.toPrecision(2));
  const pct = (v, d = 1) => fin(v) ? (100 * v).toFixed(d) + ' %' : '–';
  const sorted = a => a.filter(fin).sort((x, y) => x - y);
  const qtl = (s, q) => { if (!s.length) return NaN; const p = (s.length - 1) * q, i = Math.floor(p); return s[i] + (s[Math.min(i + 1, s.length - 1)] - s[i]) * (p - i) };
  const median = a => qtl(sorted(a), .5);
  const theme = () => document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  const rgba = (hex, a) => { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})` };
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const tok = () => ({ ink: css('--ink'), ink2: css('--ink-2'), ink3: css('--ink-3'), ink4: css('--ink-4'), rule: css('--rule'),
    faint: css('--rule-faint'), surface: css('--surface'), paper: css('--paper'),
    font: css('--font-sans') || 'Helvetica, Arial, sans-serif', mono: css('--font-mono') || 'Menlo, monospace' });
  function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296) }
  function perm(n, seed = 7) { const r = rng(seed), p = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [p[i], p[j]] = [p[j], p[i]] } return p }
  function sample(a, k, seed = 3) { if (a.length <= k) return a; const p = perm(a.length, seed); return p.slice(0, k).map(i => a[i]) }
  const NT = g => String(g).replace('non-targeting-', 'NT-');

  /* ── names: short on the chips, a little longer on the axes ── */
  const LABEL = { total_umi: 'UMIs', n_genes: 'Genes', pct_mt: 'Mito UMI %', pct_top50: 'Top-50 UMI %', complexity: 'Complexity',
    S_score: 'S score', G2M_score: 'G2/M score', ifn_score: 'IFN score', stress_score: 'Stress score' };
  const AXIS = { total_umi: 'UMIs per cell', n_genes: 'genes per cell', pct_mt: 'mito UMI %', pct_top50: 'top-50 UMI %', complexity: 'complexity',
    S_score: 'S score', G2M_score: 'G2/M score', ifn_score: 'IFN score', stress_score: 'stress score' };
  const MAIN_METRICS = ['total_umi', 'n_genes', 'pct_mt', 'pct_top50', 'complexity'];
  const MORE_METRICS = ['S_score', 'G2M_score', 'ifn_score', 'stress_score'];
  const LOGGABLE = new Set(['total_umi', 'n_genes']);

  /* ── data and the library ── */
  const cache = {};
  function getJSON(f) {
    if (!cache[f]) {
      cache[f] = fetch(BASE + f).then(r => r.ok ? r.json() : Promise.reject(new Error(`${f}: HTTP ${r.status}`)));
      cache[f].catch(() => { delete cache[f] });
    }
    return cache[f];
  }
  let plotlyP = null;
  function plotly() {
    if (window.Plotly) return Promise.resolve(window.Plotly);
    if (!plotlyP) plotlyP = new Promise((ok, no) => {
      const s = document.createElement('script');
      s.src = PLOTLY; s.async = true;
      if (SRI) { s.integrity = SRI; s.crossOrigin = 'anonymous' }
      s.onload = () => ok(window.Plotly);
      s.onerror = () => { plotlyP = null; no(new Error('the plotting library did not load')) };
      document.head.appendChild(s);
    });
    return plotlyP;
  }
  plotly().catch(() => {});

  /* ── identity: one colour and marker per context, from the palette ── */
  function ent(id) {
    const c = (PAL.contexts || {})[id];
    if (!c) return { id, color: '#888', symbol: 'circle', label: String(id), short: String(id), round: '' };
    return { id, color: c[theme()], symbol: c.marker, label: 'context ' + id, short: id, round: c.round };
  }
  const sw = id => { const e = ent(id); return `<i class="cc-sw${e.symbol === 'diamond' ? ' cc-sw--dia' : ''}" style="background:${e.color}"></i>` };
  const dot = c => `<i class="cc-sw" style="background:${c}"></i>`;
  const seqCols = () => { const s = (PAL.sequential || ['#9ec5f4', '#0d366b']).slice(); if (theme() === 'dark') s.reverse(); return s };
  const seqScale = () => { const s = seqCols(); return s.map((c, i) => [i / (s.length - 1), c]) };
  const phaseCol = p => ((PAL.phase || {})[theme()] || {})[p] || '#999';
  const clusterCol = i => (PAL.clusters || [])[i % ((PAL.clusters || []).length || 1)] || '#999';

  /* ── plotting ── */
  function merge(a, b) { for (const k in b) { if (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k] && typeof a[k] === 'object') merge(a[k], b[k]); else a[k] = b[k] } return a }
  function layout(extra = {}) {
    const t = tok(), ax = () => ({ gridcolor: t.faint, zerolinecolor: t.rule, linecolor: t.rule, tickcolor: t.rule, automargin: true,
      tickfont: { family: t.mono, size: 11, color: t.ink3 }, title: { font: { size: 12, color: t.ink3 } } });
    return merge({ paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: t.font, size: 12, color: t.ink2 },
      margin: { l: 54, r: 14, t: 28, b: 44 }, xaxis: ax(), yaxis: ax(), hovermode: 'closest', showlegend: false,
      hoverlabel: { bgcolor: t.surface, bordercolor: t.rule, font: { family: t.mono, color: t.ink, size: 12 } },
      modebar: { bgcolor: 'rgba(0,0,0,0)', color: t.ink4, activecolor: t.ink } }, extra);
  }
  const CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines'],
    toImageButtonOptions: { format: 'svg', filename: 'sidechain-vcc2026-control-cells' } };
  /* Tick labels for a log10 axis in the measure's own units: 10k, 20k rather than 4, 4.3. */
  function logTicks(lo, hi, steps = [1, 2, 5]) {
    const vals = [], text = [];
    for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) for (const m of steps) { const v = Math.log10(m) + e; if (v >= lo - .05 && v <= hi + .05) { vals.push(v); text.push(short(10 ** v)) } }
    return { tickvals: vals, ticktext: text };
  }

  /* ── the sections: each draws into its own body, with its own generation counter ── */
  const SECS = {};
  document.querySelectorAll('.cc-sec[data-sec]').forEach(el => { SECS[el.dataset.sec] = { el, body: el.querySelector('.cc-body'), gen: 0, live: false } });
  async function draw(sec, g, sel, traces, lay) {
    const P = await plotly();
    if (g !== sec.gen) return null;
    const el = typeof sel === 'string' ? sec.body.querySelector(sel) : sel;
    if (!el) return null;
    await P.react(el, traces, layout(lay), CONFIG);
    return el;
  }
  /* Replace a section's markup, keeping every plot whose id survives so a chip redraws its chart in place; plots that
     are gone are purged (their WebGL contexts are limited). */
  function swap(sec, html) {
    const t = document.createElement('div');
    t.innerHTML = html;
    t.querySelectorAll('.cc-plot[id]').forEach(n => {
      const old = sec.body.querySelector('#' + CSS.escape(n.id));
      if (old && old.classList.contains('js-plotly-plot')) { old.className = n.className + ' js-plotly-plot'; n.replaceWith(old) }
    });
    if (window.Plotly) sec.body.querySelectorAll('.js-plotly-plot').forEach(p => { if (!t.contains(p)) window.Plotly.purge(p) });
    sec.body.replaceChildren(...t.childNodes);
  }
  async function render(name) {
    const sec = SECS[name]; if (!sec) return;
    const g = ++sec.gen; sec.live = true;
    try {
      if (!IDX) { IDX = await getJSON('index.json'); PAL = IDX.palette || {} }
      if (g !== sec.gen) return;
      await RENDER[name](sec, g);
    } catch (e) {
      if (g === sec.gen) swap(sec, `<p class="cc-err">This figure could not be drawn: ${esc(e && e.message || e)}</p>`);
    }
  }

  /* ── controls ── */
  const chipBtn = (set, o, cur) => `<button type="button" class="cc-chip" data-set="${set}" data-v="${esc(o.v)}" aria-pressed="${String(o.v) === String(cur)}"${o.title ? ` title="${esc(o.title)}"` : ''}>${o.sw || ''}${esc(o.t)}</button>`;
  const chips = (set, label, opts, cur) => `<div class="cc-set" role="group" aria-label="${esc(label)}"><span class="cc-micro">${esc(label)}</span>${opts.map(o => chipBtn(set, o, cur)).join('')}</div>`;
  /* the less-used options fold under one chip; it names the option when one of them is on */
  const more = (set, opts, cur) => { const on = opts.find(o => o.v === cur);
    return `<details class="cc-more"><summary class="cc-chip" aria-pressed="${!!on}">${on ? esc(on.t) : 'More filters'}<span class="cc-caret" aria-hidden="true">▾</span></summary><div class="cc-more-pop">${opts.map(o => chipBtn(set, o, cur)).join('')}</div></details>` };
  const ctl = (...parts) => `<div class="cc-ctl">${parts.join('')}</div>`;
  const fold = (key, title, inner) => `<details class="cc-fold" data-fold="${key}"${S[key] ? ' open' : ''}><summary>${esc(title)}</summary>${inner}</details>`;
  const allCtx = () => Object.keys(IDX.contexts);
  const GROUPS = ['val', 'test', 'all'];
  const roundCtx = r => allCtx().filter(c => r === 'all' || (r === 'val' ? ent(c).round === 'validation' : ent(c).round === 'test'));
  const ctxChips = (set, cur) => chips(set, 'context', allCtx().map(c => ({ v: c, t: c, sw: sw(c) })), cur);
  const cellsOf = ctxs => Promise.all(ctxs.map(c => getJSON(`cells_${c}.json`)));
  const metricsOn = () => Object.keys(IDX.metrics);

  /* ── 01 · maps and QC ── */
  /* The cells on show, in the joint map's order or one context's own: where each sits, which context it is from, its
     row in that context's cells file, and its joint cluster. */
  function placed(J, E1, ctxs, joint, key, a, resid) {
    const X = [], Y = [], ctxOf = [], idx = [], jcl = [];
    if (joint) {
      const want = new Set(ctxs.map(c => J.contexts.indexOf(c))), seen = {}, Em = key === 'umap' ? J.umap : J.pca;
      for (let i = 0; i < J.ctx_codes.length; i++) {
        const k = J.ctx_codes[i], c = J.contexts[k]; seen[c] = seen[c] || 0;
        if (want.has(k)) { X.push(Em[a][i]); Y.push(Em[a + 1][i]); ctxOf.push(c); idx.push(seen[c]); jcl.push(J.cluster[i]) }
        seen[c]++;
      }
      return { X, Y, ctxOf, idx, jcl, vr: J.pca_var };
    }
    const Em = E1[key + (resid ? '_resid' : '')], n = Em[0].length;
    return { X: Em[a], Y: Em[a + 1], ctxOf: new Array(n).fill(ctxs[0]), idx: Array.from({ length: n }, (_, i) => i), jcl: null,
      vr: resid ? E1.pca_var_resid : E1.pca_var };
  }
  async function renderMap(sec, g) {
    const all = allCtx(), metrics = metricsOn();
    if (!GROUPS.includes(S.cells) && !all.includes(S.cells)) S.cells = 'all';
    const joint = GROUPS.includes(S.cells), ctxs = joint ? roundCtx(S.cells) : [S.cells];
    if (!['umap', 'pca12', 'pca34', 'violin'].includes(S.plot)) S.plot = 'umap';
    const vio = S.plot === 'violin';
    const grpOpts = [...(joint ? [{ v: 'context', t: 'Context' }] : []), { v: 'cluster', t: 'Cluster' }, { v: 'phase', t: 'Phase' }];
    const isGrp = v => grpOpts.some(o => o.v === v);
    const mOpt = m => ({ v: m, t: LABEL[m] || m, title: IDX.metrics[m] });
    const mainM = MAIN_METRICS.filter(m => metrics.includes(m)).map(mOpt), moreM = MORE_METRICS.filter(m => metrics.includes(m)).map(mOpt);
    if (!metrics.includes(S.measure)) S.measure = 'total_umi';
    if (!isGrp(S.by)) S.by = joint ? 'context' : 'cluster';
    if (!isGrp(S.color) && !metrics.includes(S.color)) S.color = 'cluster';
    const colour = S.color, grp = vio ? S.by : colour, qc = vio ? S.measure : colour;
    const resid = !joint && !vio && S.resid === 'on';
    const grpSet = vio ? 'by' : 'color', qcSet = vio ? 'measure' : 'color';
    const cellOpts = [{ v: 'val', t: 'Validation', title: 'contexts A, B and C on one map' }, { v: 'test', t: 'Test', title: 'contexts D, E and F on one map' },
      { v: 'all', t: 'All six', title: 'the six contexts on one map' }, ...all.map(c => ({ v: c, t: c, sw: sw(c), title: `context ${c} on its own map` }))];
    const residChip = !joint && !vio ? `<span class="cc-sep" aria-hidden="true"></span><button type="button" class="cc-chip" data-set="resid" data-v="${S.resid === 'on' ? 'off' : 'on'}" aria-pressed="${S.resid === 'on'}" title="the map after depth and the two cell-cycle scores are regressed out of every variable gene">Regressed</button>` : '';
    const showCl = grp === 'cluster';
    swap(sec, ctl(chips('cells', 'context', cellOpts, S.cells)) +
      ctl(chips('plot', 'plot', [{ v: 'umap', t: 'UMAP' }, { v: 'pca12', t: 'PC1·2' }, { v: 'pca34', t: 'PC3·4' }, { v: 'violin', t: 'Violin' }], S.plot).replace(/<\/div>$/, residChip + '</div>'),
        chips(grpSet, vio ? 'by' : 'colour', grpOpts, grp)) +
      ctl(`<div class="cc-set" role="group" aria-label="QC"><span class="cc-micro">QC</span>${mainM.map(o => chipBtn(qcSet, o, qc)).join('')}${more(qcSet, moreM, qc)}</div>`) +
      `<figure class="cc-fig cc-fig--lead"><div class="cc-key" id="cc-map-key"></div><div class="cc-plot cc-plot--map" id="${vio ? 'cc-vio' : 'cc-emb'}"></div></figure>
      <div class="cc-meta" id="cc-emb-meta"></div>` +
      (showCl ? fold('fold_cl', 'Cluster table', '<div class="cc-scroll" id="cc-emb-cl"></div>') : ''));
    const key = vio || S.plot === 'umap' ? 'umap' : 'pca', a = S.plot === 'pca34' ? 2 : 0;
    const needJ = joint && (!vio || grp === 'cluster');
    const needCells = vio || !(joint && (colour === 'context' || colour === 'cluster')) || showCl;
    const [J, E1, cells] = await Promise.all([needJ ? getJSON('joint.json') : null, joint || vio ? null : getJSON(`emb_${S.cells}.json`),
      needCells ? cellsOf(ctxs) : null]);
    if (g !== sec.gen) return;
    let P;
    if (needJ || !joint && !vio) P = placed(J, E1, ctxs, joint, key, a, resid);
    else {  // the violins of a joint view grouped by context or phase need no positions
      const ctxOf = [], idx = [];
      cells.forEach(C => { for (let i = 0; i < C.n; i++) { ctxOf.push(C.context); idx.push(i) } });
      P = { ctxOf, idx, jcl: null };
    }
    const { ctxOf, idx } = P, n = ctxOf.length;
    const B = cells ? Object.fromEntries(cells.map(c => [c.context, c])) : null;
    const cl = P.jcl ? (i => P.jcl[i]) : (i => B[ctxOf[i]].cat[resid ? 'cluster_resid' : 'cluster'].codes[idx[i]]);
    const ph = i => ['G1', 'S', 'G2M'][B[ctxOf[i]].cat.phase.codes[idx[i]]];
    const keyEl = sec.body.querySelector('#cc-map-key');
    if (!vio) {
      const { X, Y, vr } = P, p = perm(n, 5), sz = n > 60000 ? 2.2 : 3, symOf = i => joint ? ent(ctxOf[i]).symbol : 'circle';
      let marker, text, keyHtml;
      if (colour === 'context') {
        marker = { color: p.map(i => ent(ctxOf[i]).color), symbol: p.map(i => ent(ctxOf[i]).symbol), opacity: .6 };
        text = p.map(i => 'context ' + ctxOf[i]);
        keyHtml = '';  // the context chips above carry each context's colour and shape
      } else if (colour === 'cluster') {
        marker = { color: p.map(i => clusterCol(cl(i))), symbol: p.map(symOf), opacity: .7 };
        text = p.map(i => `context ${ctxOf[i]} · cluster ${cl(i)}`);
        keyHtml = '';  // a cluster's number says nothing on its own; the hover names it, the cluster table describes it
      } else if (colour === 'phase') {
        marker = { color: p.map(i => phaseCol(ph(i))), symbol: p.map(symOf), opacity: .7 };
        text = p.map(i => `context ${ctxOf[i]} · ${ph(i)}`);
        keyHtml = `<span class="cc-key-h">phase</span>` + ['G1', 'S', 'G2M'].map(l => `<span>${dot(phaseCol(l))}${l === 'G2M' ? 'G2/M' : l}</span>`).join('');
      } else {
        const V = Array.from({ length: n }, (_, i) => B[ctxOf[i]].metrics[colour][idx[i]]), s = sorted(V);
        const lo = qtl(s, .01), hi = qtl(s, .99);
        marker = { color: p.map(i => V[i]), colorscale: seqScale(), cmin: lo, cmax: hi, showscale: false, symbol: p.map(symOf), opacity: .75 };
        text = p.map(i => `context ${ctxOf[i]} · ${fmt(V[i])}`);
        keyHtml = `<span class="cc-key-h">${esc(LABEL[colour])}</span><span class="cc-key-v">${short(lo)}</span><i class="cc-ramp" style="background:linear-gradient(90deg,${seqCols().join(',')})"></i><span class="cc-key-v">${short(hi)}</span>`;
      }
      keyEl.innerHTML = keyHtml;
      const axn = S.plot === 'umap' ? ['UMAP 1', 'UMAP 2'] : [`PC${a + 1} (${pct(vr[a])})`, `PC${a + 2} (${pct(vr[a + 1])})`];
      await draw(sec, g, '#cc-emb', [{ type: 'scattergl', mode: 'markers', hoverinfo: 'text', x: p.map(i => X[i]), y: p.map(i => Y[i]), text,
        marker: { size: sz, ...marker } }], { xaxis: { title: { text: axn[0], standoff: 4 }, showgrid: false, zeroline: false },
        yaxis: { title: { text: axn[1], standoff: 4 }, showgrid: false, zeroline: false }, margin: { l: 44, r: 6, t: 4, b: 38 } });
    } else {
      // the violins: one QC measure for the same cells, one violin per group
      const vm = S.measure, logy = LOGGABLE.has(vm);
      const groupOf = grp === 'context' ? (i => ctxOf[i]) : grp === 'cluster' ? (i => String(cl(i))) : ph;
      const colOf = grp === 'context' ? (k => ent(k).color) : grp === 'cluster' ? (k => clusterCol(+k)) : phaseCol;
      const vals = {};
      for (let i = 0; i < n; i++) { const v = B[ctxOf[i]].metrics[vm][idx[i]]; if (!fin(v) || (logy && v <= 0)) continue; const k = groupOf(i); (vals[k] = vals[k] || []).push(logy ? Math.log10(v) : v) }
      const order = grp === 'context' ? ctxs.filter(c => vals[c]) : grp === 'phase' ? ['G1', 'S', 'G2M'].filter(k => vals[k]) : Object.keys(vals).sort((x, y) => x - y);
      const vt = order.map(k => { const c = colOf(k); return { type: 'violin', y: sample(vals[k], 6000), name: k === 'G2M' ? 'G2/M' : k, line: { color: c, width: 1.2 }, fillcolor: rgba(c, .25),
        box: { visible: true, width: .2, fillcolor: rgba(c, .55), line: { color: c } }, meanline: { visible: false }, points: false, spanmode: 'hard',
        hoverinfo: 'name', scalemode: 'width', width: .8 } });
      const pooled = sorted([].concat(...order.map(k => vals[k])));
      const yax = logy ? logTicks(qtl(pooled, .001), qtl(pooled, .999)) : {};
      keyEl.innerHTML = `<span class="cc-key-h">${esc(AXIS[vm])}</span><span>one violin per ${grp}, the box its middle half</span>`;
      await draw(sec, g, '#cc-vio', vt, { yaxis: { ...yax, zeroline: false }, xaxis: { type: 'category', title: { text: grp, standoff: 4 }, tickfont: { size: order.length > 8 ? 10 : 11 } },
        violingap: .2, margin: { l: 44, r: 6, t: 4, b: 38 } });
    }
    if (g !== sec.gen) return;
    sec.body.querySelector('#cc-emb-meta').textContent = `${fmt(n)} cells · ` + (joint ? (ctxs.length === all.length ? 'the six contexts' : `contexts ${ctxs.join(', ')}`) +
      (vio ? '' : ' on one map') : `context ${S.cells}` + (vio ? '' : ` on its own map${resid ? ', depth and cycle regressed out' : ''}`));
    const clEl = sec.body.querySelector('#cc-emb-cl');
    if (clEl) clEl.innerHTML = clusterTable(n, B, cl, ctxOf, idx);
  }
  function clusterTable(n, B, cl, ctxOf, idx) {
    const groups = {}, lineN = {};
    for (let i = 0; i < n; i++) { const k = cl(i); (groups[k] = groups[k] || []).push(i); lineN[ctxOf[i]] = (lineN[ctxOf[i]] || 0) + 1 }
    const mv = (ix, m) => median(ix.map(i => (B[ctxOf[i]].metrics[m] || [])[idx[i]]));
    const rows = Object.keys(groups).map(Number).sort((x, y) => x - y).map(k => {
      const ix = groups[k], cc = {}; ix.forEach(i => cc[ctxOf[i]] = (cc[ctxOf[i]] || 0) + 1);
      const c = Object.keys(cc).sort((x, y) => cc[y] - cc[x])[0];
      const ph = {}, gd = {};
      ix.forEach(i => { const C = B[ctxOf[i]], q = C.cat.phase.levels[C.cat.phase.codes[idx[i]]]; ph[q] = (ph[q] || 0) + 1; const gg = C.cat.guide.codes[idx[i]]; gd[gg] = (gd[gg] || 0) + 1 });
      const gmax = Math.max(...Object.values(gd)) / ix.length;
      return `<tr><td><span class="cc-nm">${dot(clusterCol(k))}${k}</span></td><td><span class="cc-nm">${sw(c)}${c}</span></td><td>${fmt(ix.length)}</td><td>${pct(ix.length / lineN[c])}</td><td>${fmt(mv(ix, 'total_umi'))}</td><td>${fmt(mv(ix, 'n_genes'))}</td><td>${fmt(mv(ix, 'pct_mt'), 2)}</td><td>${['G1', 'S', 'G2M'].map(q => Math.round(100 * (ph[q] || 0) / ix.length)).join(' · ')}</td><td>${Object.keys(gd).length}</td><td>${pct(gmax)}</td></tr>`;
    });
    return `<table class="cc-tbl"><thead><tr><th>cluster</th><th>context</th><th>cells</th><th>of context</th><th>UMIs</th><th>genes</th><th>mito %</th><th>G1·S·G2/M %</th><th>guides</th><th>top guide</th></tr></thead><tbody>${rows.join('')}</tbody></table>`;
  }

  /* ── 02 · guides and clones ── */
  async function renderGuides(sec, g) {
    const all = allCtx();
    if (!all.includes(S.g_ctx)) S.g_ctx = all[0];
    const e = ent(S.g_ctx), t = tok();
    swap(sec, ctl(ctxChips('g_ctx', S.g_ctx)) +
      `<div class="cc-stack"><figure class="cc-fig"><div class="cc-key"><span class="cc-key-h">each guide against 500 shuffles</span><span>${dot(e.color)}guide</span><span>${dot(t.ink)}q &lt; 0.05</span></div><div class="cc-plot cc-plot--sq" id="cc-g-sc"></div></figure>
        <figure class="cc-fig"><div class="cc-key"><span class="cc-key-h">median UMIs per guide</span></div><div class="cc-plot cc-plot--bars" id="cc-g-umi"></div></figure></div>` +
      fold('fold_g', 'Most displaced guides', '<div class="cc-scroll" id="cc-g-tbl"></div>'));
    const G = await getJSON(`guides_${S.g_ctx}.json`);
    if (g !== sec.gen) return;
    const r = G.rows, flag = r.guide.map((_, i) => r.dist_q[i] < .05 || r.dist_q_resid[i] < .05);
    await draw(sec, g, '#cc-g-sc', [{ type: 'scatter', mode: 'markers+text', x: r.dist_z, y: r.dist_z_resid, text: r.guide.map((gg, i) => flag[i] ? NT(gg) : ''), textposition: 'top center',
      textfont: { family: t.mono, size: 10, color: t.ink2 }, customdata: r.guide.map((gg, i) => [gg, r.dist_q[i], r.dist_q_resid[i], r.median_umi[i]]),
      hovertemplate: '%{customdata[0]}<br>z measured %{x:.2f} (q %{customdata[1]:.3f})<br>z regressed %{y:.2f} (q %{customdata[2]:.3f})<br>median UMIs %{customdata[3]:,.0f}<extra></extra>',
      marker: { size: 9, color: flag.map(f => f ? t.ink : e.color), opacity: flag.map(f => f ? 1 : .65), line: { color: t.paper, width: 1 } } }],
      { xaxis: { title: { text: 'distance z, measured', standoff: 4 }, zeroline: true }, yaxis: { title: { text: 'distance z, regressed', standoff: 4 }, zeroline: true }, margin: { l: 44, r: 6, t: 4, b: 38 } });
    const o = r.median_umi.map((v, i) => i).sort((x, y) => r.median_umi[x] - r.median_umi[y]);
    await draw(sec, g, '#cc-g-umi', [{ type: 'bar', x: o.map(i => NT(r.guide[i])), y: o.map(i => r.median_umi[i]), marker: { color: o.map(i => flag[i] ? t.ink : e.color), opacity: .85 },
      hovertemplate: '%{x}: %{y:,.0f} UMIs<extra></extra>' }],
      { xaxis: { tickfont: { size: 9 }, tickangle: -60 }, margin: { l: 44, r: 6, t: 4, b: 64 }, bargap: .25 });
    const ord = r.guide.map((_, i) => i).sort((x, y) => r.dist_z_resid[y] - r.dist_z_resid[x]).slice(0, 12);
    sec.body.querySelector('#cc-g-tbl').innerHTML = `<table class="cc-tbl"><thead><tr><th>guide</th><th>z measured</th><th>q</th><th>z regressed</th><th>q</th><th>cluster skew q</th><th>top cluster</th><th>expected</th><th>UMIs</th><th>mito %</th><th>G2/M</th></tr></thead><tbody>` +
      ord.map(i => `<tr><td>${esc(r.guide[i])}</td><td>${fmt(r.dist_z[i], 2)}</td><td>${fmt(r.dist_q[i], 3)}</td><td>${fmt(r.dist_z_resid[i], 2)}</td><td>${fmt(r.dist_q_resid[i], 3)}</td><td>${fmt(r.chi2_q_resid[i], 3)}</td><td>${pct(r.top_cluster_share_resid[i])}</td><td>${pct(r.top_cluster_expected_resid[i])}</td><td>${fmt(r.median_umi[i])}</td><td>${fmt(r.median_pct_mt[i], 2)}</td><td>${pct(r.share_G2M[i])}</td></tr>`).join('') + '</tbody></table>';
  }

  /* ── 03 · highest-expressed genes ── */
  async function renderGenes(sec, g) {
    const all = allCtx();
    if (!all.includes(S.gn_ctx)) S.gn_ctx = all[0];
    const e = ent(S.gn_ctx), t = tok();
    swap(sec, ctl(ctxChips('gn_ctx', S.gn_ctx)) +
      `<div class="cc-stack"><figure class="cc-fig"><div class="cc-key"><span class="cc-key-h">top 20 genes · % of each cell's UMIs</span></div><div class="cc-plot cc-plot--tall" id="cc-gn-top"></div></figure>
        <figure class="cc-fig"><div class="cc-key"><span>${dot(e.color)}variable genes</span><span>${dot(t.ink4)}other genes</span></div><div class="cc-plot cc-plot--sq" id="cc-gn-md"></div></figure></div>`);
    const [G, GI] = await Promise.all([getJSON(`genes_${S.gn_ctx}.json`), getJSON('genes.json')]);
    if (g !== sec.gen) return;
    const top = G.top20_share, names = GI.genes;
    await draw(sec, g, '#cc-gn-top', [{ type: 'box', orientation: 'h', y: top.map(r => r.gene), q1: top.map(r => r.q[1]), median: top.map(r => r.q[2]), q3: top.map(r => r.q[3]),
      lowerfence: top.map(r => r.q[0]), upperfence: top.map(r => r.q[4]), fillcolor: rgba(e.color, .25), line: { color: e.color, width: 1.3 }, hoverinfo: 'x+y', name: e.short }],
      { xaxis: { title: { text: '% of the cell\'s UMIs', standoff: 4 } }, yaxis: { autorange: 'reversed', tickfont: { size: 11 } }, margin: { l: 80, r: 6, t: 4, b: 38 } });
    const hv = G.hvg, mx = G.mean_count.map(v => Math.log10(v + 1e-4)), iH = [], iN = [];
    hv.forEach((h, i) => (h ? iH : iN).push(i));
    await draw(sec, g, '#cc-gn-md', [
      { type: 'scattergl', mode: 'markers', x: iN.map(i => mx[i]), y: iN.map(i => G.detect[i]), text: iN.map(i => names[i]), hoverinfo: 'text', marker: { size: 3, color: t.ink4, opacity: .35 } },
      { type: 'scattergl', mode: 'markers', x: iH.map(i => mx[i]), y: iH.map(i => G.detect[i]), text: iH.map(i => names[i]), hoverinfo: 'text', marker: { size: 4, color: e.color, opacity: .8 } }],
      { xaxis: { title: { text: 'mean UMIs per cell', standoff: 4 }, ...logTicks(-4, 2, [1]) }, yaxis: { title: { text: 'share of cells detecting it', standoff: 4 } }, margin: { l: 44, r: 6, t: 4, b: 38 } });
  }
  const RENDER = { map: renderMap, guides: renderGuides, genes: renderGenes };

  /* ── wiring ── */
  const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const SEC_OF = k => k.startsWith('gn_') ? 'genes' : k.startsWith('g_') ? 'guides' : 'map';
  // the rail's rows: the context on show in the map wears the accent
  const cards = [...document.querySelectorAll('.cc-card[data-ctx]')];
  const markCards = () => cards.forEach(c => c.setAttribute('aria-pressed', String(c.dataset.ctx === S.cells)));
  const redraw = k => { if (SECS[k] && SECS[k].live) render(k) };
  app.addEventListener('click', ev => {
    const chip = ev.target.closest('button[data-set]');
    if (!chip) return;
    const set = chip.dataset.set, v = chip.dataset.v;
    S[set] = v;
    if (set === 'color') { if (metricsOn().includes(v)) S.measure = v; else S.by = v }
    if (set === 'measure' || set === 'by') S.color = v;
    markCards();
    render(SEC_OF(set));
  });
  // a row shows its context in every section, and the page stays where the reader is
  cards.forEach(card => card.addEventListener('click', () => {
    const c = card.dataset.ctx;
    S.cells = c; S.g_ctx = c; S.gn_ctx = c; markCards();
    render('map'); redraw('guides'); redraw('genes');
  }));
  // back to the top: the section heads' links and the page's foot
  document.addEventListener('click', ev => {
    const up = ev.target.closest('[data-scroll-top], a[href="#top"]');
    if (!up) return;
    ev.preventDefault();
    window.scrollTo({ top: 0, behavior: calm ? 'auto' : 'smooth' });
  });
  app.addEventListener('toggle', ev => { const d = ev.target; if (d.matches && d.matches('details[data-fold]')) { S[d.dataset.fold] = d.open ? '1' : '' } }, true);
  document.addEventListener('click', ev => { document.querySelectorAll('.cc-more[open]').forEach(d => { if (!d.contains(ev.target)) d.open = false }) });
  document.addEventListener('keydown', ev => { if (ev.key === 'Escape') document.querySelectorAll('.cc-more[open]').forEach(d => { d.open = false }) });
  new MutationObserver(() => Object.keys(SECS).forEach(redraw)).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  // the map draws now; the lower sections when they come within a screen of view
  markCards();
  render('map');
  const lazy = Object.keys(SECS).filter(k => k !== 'map');
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) { io.unobserve(en.target); render(en.target.dataset.sec) } }), { rootMargin: '900px 0px' });
    lazy.forEach(k => io.observe(SECS[k].el));
  } else lazy.forEach(render);
})();
