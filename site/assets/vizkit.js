/* vizkit.js: the drawing helpers and views shared by the site's figure pages and the research dashboard.
   Usage: html = VizKit.<view>.html(data, state); on a click, VizKit.click(event, state) returns {set: {...}} (merge it
   into state and draw again), {link: {...}} (the host follows it), {copy: text}, {keep: true} (nothing to draw) or null.
   Reads the JSON a builder writes; writes nothing.
   Notes: private/code-notes/site/assets/vizkit.js.md */
(function (root) {
  'use strict';

  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fin = v => v != null && isFinite(v);
  const fmt = (v, d) => !fin(v) ? '–' : (Math.abs(v) >= 1000 || (d === undefined && Number.isInteger(+v))) ? Math.round(v).toLocaleString('en-US') :
    (+v).toFixed(d ?? (Math.abs(v) < 1 ? 3 : Math.abs(v) < 10 ? 2 : 1));
  const minus = s => String(s).replace(/-/g, '−');
  const sgn = (v, d = 3) => !fin(v) ? '–' : (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
  const colour = n => n ? `var(--viz-${n})` : 'var(--ink-4)';
  const fresh = o => JSON.parse(JSON.stringify(o));

  /* controls: a chip group is [data-vk-set=key], each chip carries data-v */
  const chips = (key, label, opts, cur) => `<span class="vk-set" data-vk-set="${esc(key)}">${label ? `<span class="vk-mi">${esc(label)}</span>` : ''}${opts.map(o =>
    `<button type="button" class="vk-chip${String(o.v) === String(cur) ? ' on' : ''}" data-v="${esc(o.v)}"${o.title ? ` title="${esc(o.title)}"` : ''}>${o.sw || ''}${esc(o.t)}</button>`).join('')}</span>`;
  const swatch = c => `<i class="vk-sw" style="--c:${c}"></i>`;
  const fold = (label, html, open) => `<details class="vk-fold"${open ? ' open' : ''}><summary>${esc(label)}</summary><div>${html}</div></details>`;
  const button = (attr, text, title) => `<button type="button" class="vk-chip" ${attr}${title ? ` title="${esc(title)}"` : ''}>${text}</button>`;
  // a button that wipes something asks twice: the first click arms it for three seconds, and a redraw keeps it armed
  let ARMED = { k: '', t: 0 };
  const armed = k => ARMED.k === k && Date.now() < ARMED.t;
  const sureButton = (k, attr, text, title) => `<button type="button" class="vk-chip${armed(k) ? ' armed' : ''}" ${attr} data-vk-sure="${esc(k)}"${title ? ` title="${esc(title)}"` : ''}>${armed(k) ? 'sure? click again' : text}</button>`;
  // a drop-down: [data-vk-menu=key] opens and shuts it; state.menu names the open one, so it stays open across draws
  const menu = (key, st, label, on, body, cls) => `<details class="vk-menu"${st.menu === key ? ' open' : ''}><summary class="vk-chip${on ? ' on' : ''}" data-vk-menu="${esc(key)}">${label} ▾</summary><div class="vk-pop${cls ? ' ' + cls : ''}">${body}</div></details>`;
  const pinCell = (id, on, at = '') => `<td class="vk-pc${at ? ' vk-fz' : ''}"${at}><input type="checkbox" class="vk-pin" data-vk-pin="${esc(id)}"${on ? ' checked' : ''} title="${on ? 'let it scroll' : 'keep it at the top'}">`;

  /* a difference cell: washed by its sign, stronger the further past its bar; ▲ ▼ past the bar */
  const washA = (d, bar) => { if (!fin(bar) || bar <= 0) return .15; const r = Math.abs(d) / bar; return Math.max(.04, r > 1 ? .5 + .5 * Math.min(1, (r - 1) / 2) : .22 * r) };
  function dcell(c, withBar) {
    if (!c || !fin(c.d)) return '<td class="vk-d vk-na">–</td>';
    const out = fin(c.bar) && Math.abs(c.d) > c.bar;
    return `<td class="vk-d ${c.d > 0 ? 'up' : c.d < 0 ? 'dn' : ''}${out ? ' out' : ''}" style="--a:${Math.round(100 * washA(c.d, c.bar))}%" title="${fin(c.bar) ? '± ' + c.bar.toFixed(3) : ''}">` +
      `${sgn(c.d)}${out ? (c.d > 0 ? ' ▲' : ' ▼') : ''}${withBar && fin(c.bar) ? `<small>± ${c.bar.toFixed(3)}</small>` : ''}</td>`;
  }

  /* ── tree: rows with a parent, drawn as an indented family tree, numbers against the parent ──
     data: {models: [{id, name, parent, change, task, dir, nights, first, date, st: {state, label, entry, board},
     readings: [{name, night, base_name, short}], edges: {<vs>: {<set>: {reading, base, base_name, vs_parent, vs_base,
     n, cells: {<column>: {d, bar}}, by: {<group>: {n, n_lines, folds, by_type, cells}}}}}}], directions: [{key, name,
     colour}], sets: [{key, label}], columns: [...], groups: [{key, label, title, count}], grp_default, nights: [{n, name,
     title, date, base}], how: {<vs>: text}, notes: [text], ref_default, gain (a host that opens a pair's knockdowns),
     ref: {ref, set, pending,
     models: {<id>: edge | {self} | {pending} | {none}}}}.
     Models come in display order: depth first, siblings as they should stand. In `ref` mode every row is measured
     against one model (data.ref, fetched by the host for state.ref); `show: 'sel'` draws only the selected rows, flat.
     An open row dims every row but the line back to its root and the row its numbers are measured against. */
  const TREE = { set: '', vs: 'parent', ref: '', show: 'all', night: '', dir: '', grp: '', rest: 'dim', open: '', shut: [], pin: [], menu: '' };
  // the state made whole: the gene set, and in `ref` mode the reference (the first selected row, else the data's default)
  function treeRef(D, st) {
    const M = D.models || [], by = Object.fromEntries(M.map(m => [m.id, m])), sets = D.sets || [];
    if (!sets.some(s => s.key === st.set)) st.set = sets.length ? sets[0].key : '';
    const gs = D.groups || [];
    if (gs.length && !gs.some(g => g.key === st.grp)) st.grp = D.grp_default || gs[0].key;
    if (st.vs !== 'ref') return '';
    if (!by[st.ref]) { const p = new Set(st.pin || []); st.ref = (M.find(m => p.has(m.id)) || by[D.ref_default] || M[0] || {}).id || '' }
    return st.ref;
  }
  function treeHTML(D, st) {
    const M = D.models || [], by = Object.fromEntries(M.map(m => [m.id, m])), dirs = Object.fromEntries((D.directions || []).map(d => [d.key, d]));
    const sets = D.sets || [], cols = D.columns || ['overall'];
    const ref = treeRef(D, st), R = ref && D.ref && D.ref.ref === ref && D.ref.set === st.set ? D.ref : null;
    const pins = new Set((st.pin || []).filter(id => by[id]));
    const sel = st.show === 'sel';
    const nightBase = (D.nights || []).find(x => String(x.n) === String(st.night));
    const hit = m => (st.night === '' || (m.nights || []).map(String).includes(String(st.night)) || (nightBase && nightBase.base === m.id)) &&
      (!st.dir || m.dir === st.dir);
    const filtered = !sel && (st.night !== '' || !!st.dir);
    const grp = (D.groups || []).find(g => g.key === st.grp);
    const inGrp = e => !e || !e.by || !grp ? e : e.by[grp.key] ? Object.assign({}, e, e.by[grp.key]) : { none: `not read on ${grp.label}` };
    const edge = m => inGrp(ref ? (m.id === ref ? { self: true } : R ? R.models[m.id] || { none: 'not in the comparison' } : { pending: true }) : ((m.edges || {})[st.vs] || {})[st.set]);
    let lin = null, cmp = ref;
    if (!sel && st.open && by[st.open]) {
      lin = new Set(); let x = by[st.open];
      while (x && !lin.has(x.id)) { lin.add(x.id); x = by[x.parent] }
      const oe = edge(by[st.open]); if (!ref && oe && oe.base) cmp = oe.base;
      if (cmp) lin.add(cmp);
    }
    const keep = new Set();
    M.forEach(m => { if (!filtered || st.rest !== 'hide' || hit(m) || pins.has(m.id) || (lin && lin.has(m.id)) || m.id === ref) { let x = m; while (x && !keep.has(x.id)) { keep.add(x.id); x = by[x.parent] } } });
    const shut = new Set(st.shut || []);
    const kids = {};
    M.forEach(m => { if (keep.has(m.id)) { const p = by[m.parent] ? m.parent : ''; (kids[p] = kids[p] || []).push(m.id) } });
    const rows = [];
    if (sel) M.forEach(m => { if (pins.has(m.id) || m.id === ref) rows.push({ m, depth: 0, lines: [], nk: 0, open: true }) });
    else {
      const walk = (id, depth, lines) => {
        const m = by[id], k = kids[id] || [], open = !shut.has(id);
        rows.push({ m, depth, lines, nk: k.length, open });
        if (open) k.forEach((c, i) => walk(c, depth + 1, [...lines, i < k.length - 1]));
      };
      (kids[''] || []).forEach(id => walk(id, 0, []));
    }
    const ns = rows.map(({ m }) => edge(m)).filter(e => e && e.cells && fin(e.n)).map(e => e.n);
    const most = ns.length ? Math.max(...ns) : 0;
    const showN = !!grp && (!!grp.count || ns.some(n => n < most));
    const unit = e => e.by_type === false ? 'lines' : 'cell types';
    const nCell = e => !showN ? '' : !e || !e.cells || !fin(e.n_lines) ? `<td class="vk-ln vk-na">${e && e.pending ? '…' : e && e.self ? '' : '–'}</td>` :
      `<td class="vk-ln${e.n < most ? ' short' : ''}" title="${esc((e.folds || []).join(', '))}${e.n < most ? ` · ${e.n} of the ${most} ${unit(e)} other rows have` : ''}">${e.n_lines}${e.by_type !== false ? `<small>${e.n} types</small>` : ''}</td>`;
    const nc = 5 + cols.length + (showN ? 1 : 0);
    const badge = s => !s || !s.label ? '' : `<span class="vk-b ${esc(s.state)}">${esc(s.label)}${(s.board || []).length ? ' · ' + esc(s.board.join(' · ')) : (s.entry || []).length ? ' · ' + esc(s.entry.join(' · ')) : ''}</span>`;
    const blank = (t, title) => cols.map(() => `<td class="vk-d vk-na"${title ? ` title="${esc(title)}"` : ''}>${t}</td>`).join('');
    const trOf = (m, depth, lines, nk, open, tray) => {
      const e = edge(m);
      const off = e && e.cells && !ref && (st.vs === 'parent' ? !e.vs_parent : !e.vs_base);
      const g = tray || sel ? '' : lines.map((more, k) => k < depth - 1 ? (more ? `<i class="vk-v" style="--k:${k}"></i>` : '') : `<i class="${more ? 'vk-tee' : 'vk-elb'}" style="--k:${k}"></i>`).join('') +
        (nk && open ? `<i class="vk-kid" style="--k:${depth}"></i>` : '');
      const dir = dirs[m.dir], c = colour(dir && dir.colour);
      const dim = !tray && ((filtered && !hit(m)) || (lin && !lin.has(m.id)));
      const kf = !tray && !sel && nk;
      const mark = tray ? '' : ref && m.id === ref ? '<span class="vk-b vs">every row minus this</span>' : cmp === m.id && st.open ? '<span class="vk-b vs">the open row minus this</span>' : '';
      const vs = !e ? '–' : e.self ? 'the reference' : e.cells ? esc(e.base_name || e.base_label) : ref ? esc((by[ref] || {}).name || ref) : '–';
      return `<tr class="vk-r${tray ? ' vk-tray' : ''}${dim ? ' vk-dim' : ''}${m.st && m.st.state === 'dropped' ? ' vk-drop' : ''}${!tray && st.open === m.id ? ' vk-on' : ''}${ref && m.id === ref ? ' vk-ref' : ''}" data-vk-row="${esc(m.id)}"${tray ? ` data-vk-jump="${esc(m.id)}"` : ''} style="--d:${tray || sel ? 0 : depth};--c:${c}">` +
        pinCell(m.id, pins.has(m.id)) + '</td>' +
        `<td class="vk-m">${g}<span class="vk-dot${kf ? ' k' : ''}${kf && !open ? ' shut' : ''}"${kf ? ` data-vk-toggle="${esc(m.id)}" title="${open ? 'fold' : 'open'} its ${nk} branch${nk === 1 ? '' : 'es'}"` : ''}></span>` +
        `<span class="vk-t"><b>${esc(m.name)}</b>${badge(m.st)}${mark}${kf && !open ? `<span class="vk-b">${nk} folded</span>` : ''}<small>${esc(m.change)}</small></span></td>` +
        `<td class="vk-tk">${esc(m.task || '')}</td><td class="vk-n" title="${esc(m.date || '')}">${m.first == null ? '–' : m.first === 0 ? 'anchor' : esc(m.first)}</td>` +
        `<td class="vk-vs${off ? ' off' : ''}" title="${e && e.cells ? esc(e.base_name || e.base_label) + ' · ' + esc(e.reading) : e && e.none ? esc(e.none) : ''}">${vs}</td>` +
        nCell(e) + (!e ? blank('–') : e.self ? blank('0') : e.pending ? blank('…', 'being worked out') : e.none ? blank('–', e.none) :
          cols.map((k, i) => dcell(e.cells ? e.cells[k] : null, i === 0)).join('')) + '</tr>';
    };
    const body = rows.map(({ m, depth, lines, nk, open }) => {
      const tr = trOf(m, depth, lines, nk, open, false);
      if (st.open !== m.id) return tr;
      const e = edge(m);
      const dir = dirs[m.dir], c = colour(dir && dir.colour);
      const rd = (m.readings || []).map(r => `<button type="button" class="vk-link" data-vk-link="reading" data-reading="${esc(r.name)}" data-night="${r.night == null ? '' : esc(r.night)}">${esc(r.short)}<small> − ${esc(r.base_name || r.base_label)}${r.night != null ? ' · ' + (r.night ? 'night ' + esc(r.night) : 'anchor') : ''}</small></button>`).join('');
      const ns = (m.nights || []).map(n => `<button type="button" class="vk-link" data-vk-link="night" data-night="${esc(n)}">${n ? 'night ' + esc(n) : 'anchor night'}</button>`).join('');
      const other = inGrp(ref ? ((m.edges || {}).parent || {})[st.set] : ((m.edges || {})[st.vs === 'parent' ? 'base' : 'parent'] || {})[st.set]);
      const ov = x => { const c = x && x.cells && x.cells[cols[0]]; return c && fin(c.d) ? `<b>${sgn(c.d)}${fin(c.bar) && Math.abs(c.d) > c.bar ? (c.d > 0 ? ' ▲' : ' ▼') : ''}</b>${fin(c.bar) ? ` ± ${c.bar.toFixed(3)}` : ''}` : '–' };
      return tr + `<tr class="vk-x"><td colspan="${nc}"><dl>` +
        (m.parent && by[m.parent] ? `<dt>parent</dt><dd>${esc(by[m.parent].name)}</dd>` : '') +
        `<dt>direction</dt><dd>${dir ? swatch(c) + esc(dir.name) : '–'}${m.task ? ' · ' + esc(m.task) : ''}</dd>` +
        `<dt>first read</dt><dd>${m.first == null ? '–' : m.first === 0 ? 'the anchor night' : 'night ' + esc(m.first)}${m.date ? ' · ' + esc(m.date) : ''}</dd>` +
        (e && e.cells ? `<dt>these numbers</dt><dd>${esc(e.short)} <span class="vk-mi">${esc(e.reading)}${e.n ? ` · mean of ${esc(e.n)} ${unit(e)}${e.folds ? ` over ${esc(e.folds.join(', '))}` : ''}` : ''}</span></dd>` : '') +
        (other && other.cells && (!e || other.reading !== e.reading) ? `<dt>minus ${esc(other.base_name || other.base_label)}</dt><dd>${esc(cols[0])} ${ov(other)} <span class="vk-mi">${esc(other.reading)} · mean of ${esc(other.n)} ${unit(other)}</span></dd>` : '') +
        `<dt>compare</dt><dd>${ref === m.id ? '<span class="vk-mi">every row is measured against this one</span>' : `<button type="button" class="vk-link" data-vk-ref="${esc(m.id)}">measure every row against this one</button>`}` +
        (D.gain && e && e.cells && e.base ? ` <button type="button" class="vk-link" data-vk-link="gain" data-a="${esc(m.id)}" data-b="${esc(e.base)}">where its difference from ${esc(e.base_name || e.base_label || e.base)} comes from, knockdown by knockdown</button>` : '') + '</dd>' +
        `<dt>readings</dt><dd>${rd || '–'}</dd>${ns ? `<dt>open</dt><dd>${ns}</dd>` : ''}</dl></td></tr>`;
    }).join('');
    const tray = sel ? '' : M.filter(m => pins.has(m.id)).map(m => trOf(m, 0, [], 0, true, true)).join('');
    const present = new Set(M.map(m => m.dir).filter(Boolean));
    const ds = (D.directions || []).filter(d => present.has(d.key));
    const nights = (D.nights || []).slice().sort((a, b) => a.n - b.n);
    const refMenu = ref ? menu('ref', st, `minus <b>${esc((by[ref] || {}).name || ref)}</b>`, true,
      M.map(m => `<button type="button" class="vk-opt vk-optb${m.id === ref ? ' on' : ''}" data-vk-ref="${esc(m.id)}">${esc(m.name)}${m.st && m.st.board && m.st.board.length ? ` <span class="vk-mi">${esc(m.st.board.join(' · '))}</span>` : ''}</button>`).join(''), 'vk-refs') : '';
    const act = `<div class="vk-act">${pins.size ? sureButton('unpin-tree', 'data-vk-unpin="1"', `clear the selection · ${pins.size}`, 'untick every row') : ''}` +
      sureButton('reset-tree', 'data-vk-reset="tree"', 'reset the view', 'every choice back to its default, the selection too') + '</div>';
    const ctl = `<div class="vk-ctl">` +
      chips('set', 'genes', sets.map(s => ({ v: s.key, t: s.label })), st.set) +
      chips('vs', 'each row minus', [{ v: 'parent', t: 'its parent' }, { v: 'base', t: 'its night\'s base' }, { v: 'ref', t: 'one model', title: 'every row against the same model, whatever night each was built on' }], st.vs) + refMenu +
      ((D.groups || []).length ? chips('grp', 'lines in the mean', D.groups.map(g => ({ v: g.key, t: g.label, title: g.title })), st.grp) : '') +
      chips('show', 'show', [{ v: 'all', t: 'every row' }, { v: 'sel', t: `the selected · ${pins.size}`, title: 'only the rows ticked on the left, against one model' }], sel ? 'sel' : 'all') +
      (sel ? '' : chips('night', 'night', [{ v: '', t: 'all' }, ...nights.map(x => ({ v: String(x.n), t: x.n ? String(x.n) : 'anchor', title: `${x.name}${x.date ? ' · ' + x.date : ''}${x.title ? ' · ' + x.title : ''}` }))], st.night) +
        chips('dir', 'direction', [{ v: '', t: 'all' }, ...ds.map(d => ({ v: d.key, t: d.name, sw: swatch(colour(d.colour)) }))], st.dir)) +
      (filtered ? chips('rest', 'the rest', [{ v: 'dim', t: 'dimmed' }, { v: 'hide', t: 'hidden' }], st.rest) : '') +
      (!sel && (st.shut || []).length ? button('data-vk-unfold="1"', 'open every branch') : '') + '</div>';
    const th = `<thead><tr><th class="vk-pc"></th><th>model</th><th>task</th><th>night</th><th title="the model each row's numbers are measured against">minus</th>${showN ? '<th class="r" title="the held-out lines behind each row\'s mean; hover a cell for their names">lines</th>' : ''}${cols.map(k => `<th class="r">${esc(k)}</th>`).join('')}</tr>${tray}</thead>`;
    const how = D.how && D.how[ref ? 'ref' : st.vs];
    const notes = [...(D.notes || [])];
    if (ref && R && R.pending) notes.unshift(`working out ${R.pending} more comparison${R.pending === 1 ? '' : 's'} against ${(by[ref] || {}).name || ref}…`);
    if (ref && R && R.error) notes.unshift(R.error);
    if (sel && !pins.size) notes.unshift('tick the box on the left of the rows to compare, then show the selected');
    return act + ctl + (how ? `<div class="vk-lgd">${esc(how)}</div>` : '') +
      `<div class="vk-scroll vk-box"><table class="vk-tree">${th}<tbody>${body}</tbody></table></div>` +
      notes.map(t => `<div class="vk-warn">${esc(t)}</div>`).join('');
  }

  /* ── cuts: a filter typed under a column's name. Numbers: > 0.6, >= 3, < -1, 2..5 or 2 to 5, = 4, none, any (several
     at once hold together: > 0.3 < 0.5, or joined by & or and); a percent column reads its numbers as percents. Words: each must appear, !word must not. Flags:
     yes or no. Returns a test, null for an empty box, or false for one it cannot read. ── */
  function cutOf(src, kind) {
    const s = String(src || '').trim().toLowerCase().replace(/[−–]/g, '-').replace(/≥/g, '>=').replace(/≤/g, '<=')
      .replace(/\bbetween\s+(-?\d*\.?\d+)\s+and\s+(-?\d*\.?\d+)/g, '$1..$2').replace(/(-?\d*\.?\d+)\s+to\s+(-?\d*\.?\d+)/g, '$1..$2')
      .replace(/^(\d*\.?\d+)\s*-\s*(-?\d*\.?\d+)$/, '$1..$2').replace(/\s*(&|\band\b)\s*/g, ',').trim();
    if (!s) return null;
    if (kind === 'flag') return /^(y|yes|1|true|●)$/.test(s) ? v => !!v : /^(n|no|0|false)$/.test(s) ? v => !v : false;
    if (kind === 'num' || kind === 'pct') {
      if (/^(none|blank|empty)$/.test(s)) return v => !fin(v);
      if (s === 'any') return fin;
      const k = kind === 'pct' ? 100 : 1, tests = [];
      for (const t of s.replace(/\s+(?=[<>=!])/g, ',').split(/\s*,\s*/).filter(Boolean)) {
        let m = t.match(/^(-?\d*\.?\d+)\s*\.\.\s*(-?\d*\.?\d+)$/);
        if (m) { const a = +m[1], b = +m[2]; tests.push(v => v * k >= Math.min(a, b) && v * k <= Math.max(a, b)); continue }
        m = t.match(/^(>=|<=|!=|>|<|=)?\s*(-?\d*\.?\d+)$/);
        if (!m) return false;
        const x = +m[2], op = m[1] || '=';
        tests.push({ '>': v => v * k > x, '>=': v => v * k >= x, '<': v => v * k < x, '<=': v => v * k <= x, '=': v => Math.abs(v * k - x) < 1e-9, '!=': v => Math.abs(v * k - x) >= 1e-9 }[op]);
      }
      return v => fin(v) && tests.every(f => f(+v));
    }
    const w = s.split(/\s+/);
    return v => { const t = String(v ?? '').toLowerCase(); return w.every(x => x[0] === '!' ? !t.includes(x.slice(1)) : t.includes(x)) };
  }

  /* ── md: a page's paragraphs, '- ' lists and inline marks; the text is escaped before any mark is read ── */
  function md(text) {
    const inline = s => esc(s).replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, '$1<i>$2</i>')
      .replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\bPMID:? ?(\d{6,9})\b/g, '<a href="https://pubmed.ncbi.nlm.nih.gov/$1/" target="_blank" rel="noopener">PMID $1</a>');
    const out = []; let list = null;
    String(text || '').split('\n').forEach(ln => {
      const m = ln.match(/^\s*[-*] (.*)$/);
      if (m) { if (!list) { list = []; out.push(list) } list.push(m[1]); return }
      list = null;
      if (ln.trim()) out.push(ln.trim()); else out.push('');
    });
    const blocks = []; let para = [];
    const flush = () => { if (para.length) { blocks.push(`<p>${inline(para.join(' '))}</p>`); para = [] } };
    out.forEach(x => { if (Array.isArray(x)) { flush(); blocks.push(`<ul>${x.map(i => `<li>${inline(i)}</li>`).join('')}</ul>`) } else if (x === '') flush(); else para.push(x) });
    flush();
    return `<div class="vk-md">${blocks.join('')}</div>`;
  }

  /* ── list: rows of one kind in a table of chosen columns: chips, menus, a search box and a filter under each
     column; rows kept at the top, painted, copied; columns widened by a drag and kept in view; a row opening a card.
     data: {columns: [{key, label, kind: gene|tag|text|num|pct|flag, digits, title, group, show, w}], groups: [{key,
     label}], rows: [{id, ...}], filters: [{key, label, options: [{v, t}], multi}], search, copy, how};
     state: LIST below. ── */
  const LIST = { q: '', f: {}, cut: {}, sort: '', desc: '', open: '', cols: null, w: {}, freeze: 1, pin: [], paint: {}, brush: 1, only: '', menu: '' };
  const PAINTS = 6, PINW = 44, DEFW = { gene: 80, text: 160, tag: 80, num: 62, pct: 58, flag: 50 };
  const isNum = c => c.kind === 'num' || c.kind === 'pct';
  const cellOf = (c, r, x = {}) => {
    const v = r[c.key], na = '<span class="vk-na">–</span>', cl = x.c ? ' ' + x.c : '', at = x.s ? ` style="${x.s}"` : '';
    if (c.kind === 'gene') return `<td class="vk-g1${cl}"${at}><b>${esc(v)}</b>${r.sub ? `<small>${esc(r.sub)}</small>` : ''}</td>`;
    if (c.kind === 'tag' || c.kind === 'text') return `<td class="vk-tag${cl}"${at} title="${esc(v ?? '')}">${v == null || v === '' ? na : esc(v)}</td>`;
    if (c.kind === 'flag') return `<td class="vk-flag${cl}"${at}>${v ? '<span class="vk-mk">●</span>' : ''}</td>`;
    if (c.kind === 'pct') return `<td class="vk-num${cl}"${at}>${fin(v) ? Math.round(100 * v) + ' %' : na}</td>`;
    return `<td class="vk-num${cl}"${at}>${fin(v) ? minus(fmt(v, c.digits)) : na}${c.mark && r[c.mark] ? ' <span class="vk-mk">●</span>' : ''}</td>`;
  };
  const rowText = (D, r) => [r.id, r.sub, ...(D.columns || []).filter(c => c.kind === 'tag' || c.kind === 'text').map(c => r[c.key])].join(' ').toLowerCase();
  const matches = (q, t) => { const w = String(q || '').toLowerCase().split(/\s+/).filter(Boolean); return w.every(x => t.includes(x)) };
  // the state made whole against this data: a stale filter value or column is dropped, the shown columns written out
  function listState(D, st) {
    const cols = D.columns || [], by = Object.fromEntries(cols.map(c => [c.key, c]));
    st.f = Object.assign({}, st.f);
    (D.filters || []).forEach(F => { const ok = new Set(F.options.map(o => String(o.v))), v = st.f[F.key];
      st.f[F.key] = F.multi ? (Array.isArray(v) ? v.filter(x => ok.has(String(x))) : []) : (v != null && ok.has(String(v)) ? v : '') });
    if (!Array.isArray(st.cols) || !st.cols.some(k => by[k])) st.cols = cols.filter(c => c.show).map(c => c.key);
    st.cols = st.cols.filter(k => by[k]);
    ['cut', 'w', 'paint'].forEach(k => { if (!st[k] || typeof st[k] !== 'object' || Array.isArray(st[k])) st[k] = {} });
    if (!Array.isArray(st.pin)) st.pin = [];
    if (!(st.brush >= 1 && st.brush <= PAINTS)) st.brush = 1;
    return { cols, by, vis: cols.filter(c => st.cols.includes(c.key)) };
  }
  function keeper(D, st, vis) {             // the open row always shows
    const cuts = vis.map(c => [c.key, cutOf((st.cut || {})[c.key], c.kind)]).filter(([, t]) => t);
    const paint = st.paint || {};
    return r => r.id === st.open || (D.filters || []).every(F => { const v = (st.f || {})[F.key];
        return F.multi ? !(Array.isArray(v) && v.length) || v.includes(String(r[F.key])) : v === '' || v == null || String(r[F.key]) === String(v) }) &&
      (!st.only || (st.only === 'any' ? !!paint[r.id] : String(paint[r.id]) === String(st.only))) &&
      cuts.every(([k, t]) => t(r[k])) && matches(st.q, rowText(D, r));
  }
  function listHTML(D, st) {
    const { cols, vis } = listState(D, st), f = st.f;
    const pins = new Set(st.pin), paint = st.paint, keep = keeper(D, st, vis);
    let rows = D.rows || [];
    const col = cols.find(c => c.key === st.sort);
    if (col) {
      const s = st.desc ? -1 : 1, num = isNum(col) || col.kind === 'flag';
      rows = rows.slice().sort((a, b) => { const x = a[col.key], y = b[col.key];
        if (x == null || x === '') return 1; if (y == null || y === '') return -1;
        return (num ? x - y : String(x).localeCompare(String(y))) * s });
    }
    const wOf = c => Math.max(34, +(st.w[c.key] || c.w || DEFW[c.kind] || 62));
    const fz = Math.max(0, Math.min(+st.freeze || 0, vis.length)), left = [];
    vis.reduce((x, c, i) => { left[i] = x; return x + wOf(c) }, PINW);
    const fzAt = i => i < fz ? { c: 'vk-fz' + (i === fz - 1 ? ' vk-fzl' : ''), s: `left:${left[i]}px` } : {};
    const width = PINW + vis.reduce((x, c) => x + wOf(c), 0);
    const rowHTML = (r, tray) => {
      const p = paint[r.id];
      return `<tr class="vk-r${tray ? ' vk-tray' : ''}${!tray && st.open === r.id ? ' vk-on' : ''}${p ? ' vk-painted' : ''}" data-vk-row="${esc(r.id)}"${tray ? ` data-vk-jump="${esc(r.id)}"` : ''}${p ? ` style="--p:${colour(p)}"` : ''}${!tray && !keep(r) ? ' hidden' : ''}>` +
        pinCell(r.id, pins.has(r.id), ' style="left:0"') + `<i class="vk-pt" data-vk-paint="${esc(r.id)}" title="${p ? 'unpaint' : 'paint it with the brush'}"></i></td>` +
        vis.map((c, i) => cellOf(c, r, fzAt(i))).join('') + '</tr>';
    };
    const nf = 1 + vis.length;
    const body = rows.map(r => rowHTML(r, false) + (st.open === r.id ? `<tr class="vk-x vk-cardrow"><td colspan="${nf}"><div class="vk-card" data-vk-card="${esc(r.id)}"><div class="vk-mi">loading…</div></div></td></tr>` : '')).join('');
    const tray = (D.rows || []).filter(r => pins.has(r.id)).map(r => rowHTML(r, true)).join('');
    const thc = (c, i, inner, sort) => { const z = fzAt(i); return `<th class="${isNum(c) ? 'r' : ''}${sort && st.sort === c.key ? ' on' : ''}${z.c ? ' ' + z.c : ''}"${z.s ? ` style="${z.s}"` : ''}${sort ? ` data-vk-sort="${esc(c.key)}" title="${esc(c.title || '')}"` : ''}>${inner}</th>` };
    const th = `<thead><tr class="vk-hr"><th class="vk-pc vk-fz" style="left:0"></th>` +
      vis.map((c, i) => thc(c, i, `<span class="vk-hl">${esc(c.label)}${st.sort === c.key ? (st.desc ? ' ↓' : ' ↑') : ''}</span>` +
        `<button type="button" class="vk-fzb${i === fz - 1 ? ' on' : ''}" data-vk-freeze="${i + 1}" title="${i === fz - 1 ? 'let these columns scroll' : 'keep the columns up to this one in view'}">⇤</button><span class="vk-rz" data-vk-rz="${esc(c.key)}" title="drag to widen"></span>`, true)).join('') + '</tr>' +
      `<tr class="vk-cuts"><th class="vk-pc vk-fz" style="left:0"></th>${vis.map((c, i) => thc(c, i, `<input class="vk-cut" data-vk-cut="${esc(c.key)}" value="${esc(st.cut[c.key] || '')}" placeholder="${isNum(c) ? '> <' : c.kind === 'flag' ? 'yes' : 'word'}" title="${isNum(c) ? '> 0.6 · < -1 · >= 3 · 0.3..0.5 (between) · > 0.3 < 0.5 · none · any' : c.kind === 'flag' ? 'yes or no' : 'a word it holds; !word for one it does not'}" spellcheck="false" autocomplete="off">`, false)).join('')}</tr>` +
      tray + '</thead>';
    const ctl = `<div class="vk-ctl vk-c1">` + (D.filters || []).map(F => {
      if (!F.multi) return chips('f:' + F.key, F.label, F.options, f[F.key] ?? '');
      const sel = f[F.key];
      return menu('f:' + F.key, st, `${esc(F.label)} · ${sel.length ? sel.length === 1 ? esc(sel[0]) : sel.length + ' of ' + F.options.length : 'all'}`, sel.length,
        `<label class="vk-opt"><input type="checkbox" data-vk-fv="${esc(F.key)}" data-v=""${sel.length ? '' : ' checked'}> all</label>` +
        F.options.map(o => `<label class="vk-opt"><input type="checkbox" data-vk-fv="${esc(F.key)}" data-v="${esc(o.v)}"${sel.includes(String(o.v)) ? ' checked' : ''}> ${esc(o.t)}</label>`).join(''));
    }).join('') +
      `<span class="vk-set"><span class="vk-mi">find</span><input type="search" class="vk-q" data-vk-q="1" value="${esc(st.q || '')}" placeholder="${esc(D.search || '')}" spellcheck="false" autocomplete="off"></span>` + '</div>';
    const groups = (D.groups || []).concat([{ key: '', label: 'other' }]), shown = new Set(st.cols);
    const colMenu = menu('cols', st, `columns · ${vis.length} of ${cols.length}`, false, groups.map(g => {
      const cs = cols.filter(c => (c.group || '') === g.key); if (!cs.length) return '';
      const all = cs.every(c => shown.has(c.key));
      return `<div class="vk-grp"><button type="button" class="vk-gh" data-vk-colgrp="${esc(cs.map(c => c.key).join(','))}" data-on="${all ? '1' : ''}" title="${all ? 'hide' : 'show'} the whole group">${esc(g.label)}</button>` +
        cs.map(c => `<label class="vk-opt" title="${esc(c.title || '')}"><input type="checkbox" data-vk-col="${esc(c.key)}"${shown.has(c.key) ? ' checked' : ''}> ${esc(c.label)}</label>`).join('') + '</div>';
    }).join('') + `<div class="vk-grp">${button(`data-vk-coldef="${esc(cols.filter(c => c.show).map(c => c.key).join(','))}"`, 'the default columns')}</div>`, 'vk-cols');
    const pc = {}; Object.values(paint).forEach(n => { pc[n] = (pc[n] || 0) + 1 });
    const np = Object.keys(paint).length;
    const tools = `<div class="vk-ctl vk-c2">` + colMenu +
      `<span class="vk-set" data-vk-set="brush"><span class="vk-mi">paint</span>${Array.from({ length: PAINTS }, (_, k) => `<button type="button" class="vk-chip vk-brush${String(st.brush) === String(k + 1) ? ' on' : ''}" data-v="${k + 1}" title="brush ${k + 1}">${swatch(colour(k + 1))}${pc[k + 1] ? `<span>${pc[k + 1]}</span>` : ''}</button>`).join('')}</span>` +
      button('data-vk-paintall="1"', 'paint the shown', 'paint every row the filters leave with the brush') +
      (np ? chips('only', 'show', [{ v: '', t: 'all' }, { v: 'any', t: 'painted' }, ...Object.keys(pc).sort().map(n => ({ v: n, t: String(pc[n]), sw: swatch(colour(n)) }))], st.only || '') : '') +
      button('data-vk-copy="1"', `copy the shown ${esc(D.copy || 'ids')}`, 'one per line, in the order shown') +
      `<span class="vk-mi vk-cnt" data-vk-count="1"></span></div>`;
    const act = `<div class="vk-act">` + (pins.size ? sureButton('unpin-list', 'data-vk-unpin="1"', `clear the selection · ${pins.size}`, 'untick every row kept at the top') : '') +
      (np ? sureButton('unpaint', 'data-vk-unpaint="1"', `clear the paint · ${np}`, 'take every colour off') : '') +
      sureButton('unfilter', 'data-vk-unfilter="1"', 'clear the filters', 'the chips, the menus, the search and every column box') +
      sureButton('reset-list', 'data-vk-reset="list"', 'reset the view', 'filters, columns, widths, sort and the selection back to their defaults; the paint stays') + '</div>';
    const hint = `<div class="vk-hint">the box under a column: <code>&gt; 0.6</code> above · <code>&lt;= -1</code> at or below · <code>0.3..0.5</code> or <code>0.3 to 0.5</code> between, both ends kept · <code>&gt; 0.3 &lt; 0.5</code> both at once · <code>none</code> blank · <code>any</code> not blank · a word in a text column · <code>yes</code> or <code>no</code> for a flag; every box holds at once</div>`;
    return `<div class="vk-listv">` + act + ctl + tools + hint + (D.how ? fold('how to read it', `<p class="vk-lgd">${esc(D.how)}</p>`) : '') +
      `<div class="vk-scroll vk-box vk-listwrap"><table class="vk-tree vk-list vk-fixed" style="width:${width}px"><colgroup><col style="width:${PINW}px">${vis.map(c => `<col data-k="${esc(c.key)}" style="width:${wOf(c)}px">`).join('')}</colgroup>${th}<tbody>${body}</tbody></table></div></div>`;
  }
  // typing in the search or a column box narrows the rows in place, so the box keeps its focus
  function listFilter(el, D, st) {
    const { by, vis } = listState(D, st), keep = keeper(D, st, vis);
    const rows = D._by || (D._by = Object.fromEntries((D.rows || []).map(r => [r.id, r])));
    let k = 0, n = 0;
    el.querySelectorAll('tbody tr[data-vk-row]').forEach(tr => { const r = rows[tr.dataset.vkRow]; if (!r) return; n++; const ok = keep(r); tr.hidden = !ok; if (ok) k++;
      const nx = tr.nextElementSibling; if (nx && nx.classList.contains('vk-cardrow')) nx.hidden = !ok });
    const c = el.querySelector('[data-vk-count]'); if (c) c.textContent = `${k} of ${n}`;
    el.querySelectorAll('[data-vk-cut]').forEach(i => { const t = cutOf(i.value, (by[i.dataset.vkCut] || {}).kind); i.classList.toggle('bad', t === false); i.classList.toggle('set', !!t) });
  }
  // the host calls this once per draw: typing filters in place, a drag on a column's right edge sets its width.
  // onState(patch, redraw) merges the patch into the host's state.
  function listBind(el, D, st, onState) {
    el.addEventListener('input', ev => {
      const q = ev.target.closest('[data-vk-q]'), c = ev.target.closest('[data-vk-cut]');
      if (q) onState({ q: q.value });
      else if (c) onState({ cut: Object.assign({}, st.cut, { [c.dataset.vkCut]: c.value }) });
      else return;
      listFilter(el, D, st);
    });
    el.addEventListener('pointerdown', ev => {
      const h = ev.target.closest('[data-vk-rz]'); if (!h) return;
      ev.preventDefault();
      const key = h.dataset.vkRz, tbl = h.closest('table'), col = tbl.querySelector(`col[data-k="${CSS.escape(key)}"]`);
      const x0 = ev.clientX, w0 = col ? parseFloat(col.style.width) : h.parentElement.offsetWidth, t0 = parseFloat(tbl.style.width) || tbl.offsetWidth;
      let w = w0;
      const mv = e2 => { w = Math.max(34, Math.round(w0 + e2.clientX - x0)); if (col) col.style.width = w + 'px'; tbl.style.width = (t0 + w - w0) + 'px' };
      const stop = e3 => { e3.stopPropagation(); e3.preventDefault() };
      const up = () => {
        removeEventListener('pointermove', mv); removeEventListener('pointerup', up); document.body.classList.remove('vk-dragging');
        addEventListener('click', stop, true); setTimeout(() => removeEventListener('click', stop, true), 0);   // the drag is not a sort
        if (w !== w0) onState({ w: Object.assign({}, st.w, { [key]: w }) }, true);
      };
      addEventListener('pointermove', mv); addEventListener('pointerup', up); document.body.classList.add('vk-dragging');
    });
  }

  function click(e, st) {
    const t0 = e.target;
    const close = st.menu && !t0.closest('.vk-menu') ? { menu: '' } : null;
    const done = s => ({ set: Object.assign({}, close, s) });
    const sure = t0.closest('[data-vk-sure]');
    if (sure && !armed(sure.dataset.vkSure)) {
      const k = sure.dataset.vkSure, was = sure.innerHTML;
      ARMED = { k, t: Date.now() + 3000 }; sure.classList.add('armed'); sure.textContent = 'sure? click again';
      setTimeout(() => { if (sure.isConnected && !armed(k)) { sure.classList.remove('armed'); sure.innerHTML = was } }, 3050);
      return { keep: true };
    }
    if (sure) ARMED = { k: '', t: 0 };
    const rf = t0.closest('[data-vk-ref]');
    if (rf) return { set: { vs: 'ref', ref: rf.dataset.vkRef, menu: '' } };
    const mn = t0.closest('[data-vk-menu]');
    if (mn) { e.preventDefault(); const k = mn.dataset.vkMenu; return { set: { menu: st.menu === k ? '' : k } } }
    const fv = t0.closest('[data-vk-fv]');
    if (fv) { const k = fv.dataset.vkFv, v = fv.dataset.v, cur = Array.isArray((st.f || {})[k]) ? st.f[k] : [];
      return { set: { f: Object.assign({}, st.f, { [k]: v === '' ? [] : cur.includes(v) ? cur.filter(x => x !== v) : [...cur, v] }) } } }
    const cl = t0.closest('[data-vk-col]');
    if (cl) { const k = cl.dataset.vkCol, cur = st.cols || []; return { set: { cols: cur.includes(k) ? cur.filter(x => x !== k) : [...cur, k] } } }
    const cg = t0.closest('[data-vk-colgrp]');
    if (cg) { const ks = cg.dataset.vkColgrp.split(','), cur = (st.cols || []).filter(x => !ks.includes(x)); return { set: { cols: cg.dataset.on ? cur : [...cur, ...ks] } } }
    const cd = t0.closest('[data-vk-coldef]');
    if (cd) return { set: { cols: cd.dataset.vkColdef.split(','), w: {} } };
    if (t0.closest('.vk-pop')) return { keep: true };   // a label's click: its box gets its own click next
    if (t0.closest('[data-vk-q], [data-vk-cut], [data-vk-rz]')) return { keep: true };
    const rs = t0.closest('[data-vk-reset]');
    if (rs) return { set: rs.dataset.vkReset === 'tree' ? fresh(TREE) : Object.assign(fresh(LIST), { paint: st.paint || {}, brush: st.brush || 1 }) };
    const pn = t0.closest('[data-vk-pin]');
    if (pn) { const s = new Set(st.pin || []), id = pn.dataset.vkPin; s.has(id) ? s.delete(id) : s.add(id); return done({ pin: [...s] }) }
    if (t0.closest('[data-vk-unpin]')) return done({ pin: [] });
    const pt = t0.closest('[data-vk-paint]');
    if (pt) { const id = pt.dataset.vkPaint, b = +st.brush || 1, P = Object.assign({}, st.paint); if (+P[id] === b) delete P[id]; else P[id] = b; return done({ paint: P }) }
    const shownIds = () => { const v = t0.closest('.vk-listv'); return v ? [...v.querySelectorAll('tbody tr[data-vk-row]:not([hidden])')].map(x => x.dataset.vkRow) : [] };
    if (t0.closest('[data-vk-paintall]')) { const b = +st.brush || 1, P = Object.assign({}, st.paint); shownIds().forEach(id => { P[id] = b }); return done({ paint: P }) }
    if (t0.closest('[data-vk-unpaint]')) return done({ paint: {}, only: '' });
    if (t0.closest('[data-vk-copy]')) { const ids = shownIds(); return { copy: ids.join('\n'), n: ids.length } }
    if (t0.closest('[data-vk-unfilter]')) return done({ f: {}, cut: {}, q: '', only: '' });
    const fz = t0.closest('[data-vk-freeze]');
    if (fz) { const n = +fz.dataset.vkFreeze; return done({ freeze: +st.freeze === n ? 0 : n }) }
    const c = t0.closest('[data-vk-set] .vk-chip');
    if (c) { const k = c.parentElement.dataset.vkSet, v = c.dataset.v;
      if (k === 'show' && v === 'sel' && st.vs !== 'ref') return done({ show: v, vs: 'ref', ref: '' });   // the selected, on one footing
      return done(k.startsWith('f:') ? { f: Object.assign({}, st.f, { [k.slice(2)]: v }) } : { [k]: v }) }
    const so = t0.closest('[data-vk-sort]');
    if (so) { const k = so.dataset.vkSort; return done(st.sort === k ? { desc: st.desc ? '' : '1' } : { sort: k, desc: '1' }) }
    if (t0.closest('[data-vk-unfold]')) return done({ shut: [] });
    const t = t0.closest('[data-vk-toggle]');
    if (t) { const id = t.dataset.vkToggle, s = new Set(st.shut || []); s.has(id) ? s.delete(id) : s.add(id); return done({ shut: [...s] }) }
    const l = t0.closest('[data-vk-link]');
    if (l) return { link: { kind: l.dataset.vkLink, reading: l.dataset.reading || '', night: l.dataset.night || '', path: l.dataset.path || '', a: l.dataset.a || '', b: l.dataset.b || '' } };
    const j = t0.closest('[data-vk-jump]');
    if (j) return Object.assign(done({ open: j.dataset.vkJump }), { jump: j.dataset.vkJump });
    if (t0.closest('.vk-x')) return close ? { set: close } : null;
    const r = t0.closest('[data-vk-row]');
    if (r) return done({ open: st.open === r.dataset.vkRow ? '' : r.dataset.vkRow });
    return close ? { set: close } : null;
  }

  /* ── card: one thing's page: {id, title, name, tags, sub, path, blocks: [{kind: md|facts|table, h, open, md | rows |
     columns, how}]} ── */
  function cardHTML(C) {
    if (!C) return '';
    if (C.error) return `<div class="vk-warn">${esc(C.error)}</div>`;
    const block = b => {
      let inner = '';
      if (b.kind === 'md') inner = md(b.md);
      else if (b.kind === 'facts') inner = `<dl class="vk-facts">${(b.rows || []).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>`;
      else if (b.kind === 'table') inner = (b.how ? `<p class="vk-lgd">${esc(b.how)}</p>` : '') + `<table class="vk-tree vk-mini"><thead><tr>${(b.columns || []).map(c => `<th class="${isNum(c) ? 'r' : ''}">${esc(c.label)}</th>`).join('')}</tr></thead><tbody>${(b.rows || []).map(r => `<tr>${(b.columns || []).map(c => cellOf(c, r)).join('')}</tr>`).join('') || `<tr><td class="vk-na" colspan="${(b.columns || []).length}">none</td></tr>`}</tbody></table>`;
      return `<details class="vk-fold vk-blk"${b.open ? ' open' : ''}><summary>${esc(b.h)}</summary><div>${inner}</div></details>`;
    };
    return `<div class="vk-chead"><h3>${esc(C.title)}${C.name ? ` <span>${esc(C.name)}</span>` : ''}</h3>${(C.tags || []).filter(Boolean).map(t => `<span class="vk-b">${esc(t)}</span>`).join('')}` +
      `${C.sub ? `<div class="vk-mi">${esc(C.sub)}</div>` : ''}${C.path ? `<button type="button" class="vk-link" data-vk-link="path" data-path="${esc(C.path)}">${esc(C.path)}</button>` : ''}</div>` +
      `<div class="vk-blocks">${(C.blocks || []).map(block).join('')}</div>`;
  }

  root.VizKit = { esc, fin, fmt, sgn, minus, chips, swatch, fold, dcell, colour, click, md, cut: cutOf,
    tree: { html: treeHTML, ref: treeRef, defaults: TREE }, list: { html: listHTML, filter: listFilter, bind: listBind, defaults: LIST }, card: { html: cardHTML } };
})(typeof window !== 'undefined' ? window : this);
