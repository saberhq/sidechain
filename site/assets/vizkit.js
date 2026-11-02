/* vizkit.js: the drawing helpers and views shared by the site's figure pages and the research dashboard.
   Usage: html = VizKit.<view>.html(data, state); on a click, VizKit.click(event, state) returns {set: {...}} (merge it
   into state and draw again), {link: {...}} (the host follows it) or null.
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

  /* controls: a chip group is [data-vk-set=key], each chip carries data-v */
  const chips = (key, label, opts, cur) => `<span class="vk-set" data-vk-set="${esc(key)}">${label ? `<span class="vk-mi">${esc(label)}</span>` : ''}${opts.map(o =>
    `<button type="button" class="vk-chip${String(o.v) === String(cur) ? ' on' : ''}" data-v="${esc(o.v)}"${o.title ? ` title="${esc(o.title)}"` : ''}>${o.sw || ''}${esc(o.t)}</button>`).join('')}</span>`;
  const swatch = c => `<i class="vk-sw" style="--c:${c}"></i>`;
  const fold = (label, html, open) => `<details class="vk-fold"${open ? ' open' : ''}><summary>${esc(label)}</summary><div>${html}</div></details>`;

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
     readings: [{name, night, base_label, short}], edges: {<vs>: {<set>: {reading, base_label, vs_parent, vs_base,
     n, cells: {<column>: {d, bar}}}}}}], directions: [{key, name, colour}], sets: [{key, label}], columns: [...],
     nights: [{n, name, title, date, base}], how: {<vs>: text}, unlisted: [{model}], problems: [...]}.
     Models come in display order: depth first, siblings as they should stand. */
  const TREE = { set: '', vs: 'parent', night: '', dir: '', rest: 'dim', open: '', shut: [] };
  function treeHTML(D, st) {
    const M = D.models || [], by = Object.fromEntries(M.map(m => [m.id, m])), dirs = Object.fromEntries((D.directions || []).map(d => [d.key, d]));
    const sets = D.sets || [], cols = D.columns || ['overall'];
    if (!sets.some(s => s.key === st.set)) st.set = sets.length ? sets[0].key : '';
    const nightBase = (D.nights || []).find(x => String(x.n) === String(st.night));
    const hit = m => (st.night === '' || (m.nights || []).map(String).includes(String(st.night)) || (nightBase && nightBase.base === m.id)) &&
      (!st.dir || m.dir === st.dir);
    const filtered = st.night !== '' || !!st.dir;
    const keep = new Set();
    M.forEach(m => { if (!filtered || st.rest !== 'hide' || hit(m)) { let x = m; while (x && !keep.has(x.id)) { keep.add(x.id); x = by[x.parent] } } });
    const shut = new Set(st.shut || []);
    const kids = {};
    M.forEach(m => { if (keep.has(m.id)) { const p = by[m.parent] ? m.parent : ''; (kids[p] = kids[p] || []).push(m.id) } });
    const rows = [];
    const walk = (id, depth, lines) => {
      const m = by[id], k = kids[id] || [], open = !shut.has(id);
      rows.push({ m, depth, lines, nk: k.length, open });
      if (open) k.forEach((c, i) => walk(c, depth + 1, [...lines, i < k.length - 1]));
    };
    (kids[''] || []).forEach(id => walk(id, 0, []));
    const nc = 3 + cols.length;
    const badge = s => !s || !s.label ? '' : `<span class="vk-b ${esc(s.state)}">${esc(s.label)}${(s.board || []).length ? ' · ' + esc(s.board.join(' · ')) : (s.entry || []).length ? ' · ' + esc(s.entry.join(' · ')) : ''}</span>`;
    const body = rows.map(({ m, depth, lines, nk, open }) => {
      const e = ((m.edges || {})[st.vs] || {})[st.set];
      const off = e && (st.vs === 'parent' ? !e.vs_parent : !e.vs_base);
      const g = lines.map((more, k) => k < depth - 1 ? (more ? `<i class="vk-v" style="--k:${k}"></i>` : '') : `<i class="${more ? 'vk-tee' : 'vk-elb'}" style="--k:${k}"></i>`).join('') +
        (nk && open ? `<i class="vk-kid" style="--k:${depth}"></i>` : '');
      const dir = dirs[m.dir], c = colour(dir && dir.colour);
      const dim = filtered && !hit(m);
      const tr = `<tr class="vk-r${dim ? ' vk-dim' : ''}${m.st && m.st.state === 'dropped' ? ' vk-drop' : ''}${st.open === m.id ? ' vk-on' : ''}" data-vk-row="${esc(m.id)}" style="--d:${depth};--c:${c}">` +
        `<td class="vk-m">${g}<span class="vk-dot${nk ? ' k' : ''}${nk && !open ? ' shut' : ''}"${nk ? ` data-vk-toggle="${esc(m.id)}" title="${open ? 'fold' : 'open'} its ${nk} branch${nk === 1 ? '' : 'es'}"` : ''}></span>` +
        `<span class="vk-t"><b>${esc(m.name)}</b>${badge(m.st)}${nk && !open ? `<span class="vk-b">${nk} folded</span>` : ''}<small>${esc(m.change)}${off ? ` <em>· against ${esc(e.base_label)}</em>` : ''}</small></span></td>` +
        `<td class="vk-tk">${esc(m.task || '')}</td><td class="vk-n" title="${esc(m.date || '')}">${m.first == null ? '–' : m.first === 0 ? 'anchor' : esc(m.first)}</td>` +
        cols.map((k, i) => dcell(e && e.cells ? e.cells[k] : null, i === 0)).join('') + '</tr>';
      if (st.open !== m.id) return tr;
      const rd = (m.readings || []).map(r => `<button type="button" class="vk-link" data-vk-link="reading" data-reading="${esc(r.name)}" data-night="${r.night == null ? '' : esc(r.night)}">${esc(r.short)}<small> − ${esc(r.base_label)}${r.night != null ? ' · ' + (r.night ? 'night ' + esc(r.night) : 'anchor') : ''}</small></button>`).join('');
      const ns = (m.nights || []).map(n => `<button type="button" class="vk-link" data-vk-link="night" data-night="${esc(n)}">${n ? 'night ' + esc(n) : 'anchor night'}</button>`).join('');
      const other = ((m.edges || {})[st.vs === 'parent' ? 'base' : 'parent'] || {})[st.set];
      const ov = x => { const c = x && x.cells && x.cells[cols[0]]; return c && fin(c.d) ? `<b>${sgn(c.d)}${fin(c.bar) && Math.abs(c.d) > c.bar ? (c.d > 0 ? ' ▲' : ' ▼') : ''}</b>${fin(c.bar) ? ` ± ${c.bar.toFixed(3)}` : ''}` : '–' };
      return tr + `<tr class="vk-x"><td colspan="${nc}"><dl>` +
        (m.parent && by[m.parent] ? `<dt>parent</dt><dd>${esc(by[m.parent].name)}</dd>` : '') +
        `<dt>direction</dt><dd>${dir ? swatch(c) + esc(dir.name) : '–'}${m.task ? ' · ' + esc(m.task) : ''}</dd>` +
        `<dt>first read</dt><dd>${m.first == null ? '–' : m.first === 0 ? 'the anchor night' : 'night ' + esc(m.first)}${m.date ? ' · ' + esc(m.date) : ''}</dd>` +
        (e ? `<dt>these numbers</dt><dd>${esc(e.short)} <span class="vk-mi">${esc(e.reading)} · mean of ${esc(e.n)} cell types</span></dd>` : '') +
        (other && (!e || other.reading !== e.reading) ? `<dt>against ${esc(other.base_label)}</dt><dd>${esc(cols[0])} ${ov(other)} <span class="vk-mi">${esc(other.reading)} · mean of ${esc(other.n)} cell types</span></dd>` : '') +
        `<dt>readings</dt><dd>${rd || '–'}</dd>${ns ? `<dt>open</dt><dd>${ns}</dd>` : ''}</dl></td></tr>`;
    }).join('');
    const present = new Set(M.map(m => m.dir).filter(Boolean));
    const ds = (D.directions || []).filter(d => present.has(d.key));
    const nights = (D.nights || []).slice().sort((a, b) => a.n - b.n);
    const ctl = `<div class="vk-ctl">` +
      chips('set', 'genes', sets.map(s => ({ v: s.key, t: s.label })), st.set) +
      chips('vs', 'each row minus', [{ v: 'parent', t: 'its parent' }, { v: 'base', t: 'its night\'s base' }], st.vs) +
      chips('night', 'night', [{ v: '', t: 'all' }, ...nights.map(x => ({ v: String(x.n), t: x.n ? String(x.n) : 'anchor', title: `${x.name}${x.date ? ' · ' + x.date : ''}${x.title ? ' · ' + x.title : ''}` }))], st.night) +
      chips('dir', 'direction', [{ v: '', t: 'all' }, ...ds.map(d => ({ v: d.key, t: d.name, sw: swatch(colour(d.colour)) }))], st.dir) +
      (filtered ? chips('rest', 'the rest', [{ v: 'dim', t: 'dimmed' }, { v: 'hide', t: 'hidden' }], st.rest) : '') +
      ((st.shut || []).length ? `<button type="button" class="vk-chip" data-vk-unfold="1">open every branch</button>` : '') + '</div>';
    const th = `<thead><tr><th>model</th><th>task</th><th>night</th>${cols.map(k => `<th class="r">${esc(k)}</th>`).join('')}</tr></thead>`;
    const notes = [(D.unlisted || []).length ? `read on a night, not on the list: ${D.unlisted.map(u => esc(u.model)).join(', ')}` : '',
      ...(D.problems || []).map(esc)].filter(Boolean);
    return ctl + (D.how && D.how[st.vs] ? `<div class="vk-lgd">${esc(D.how[st.vs])}</div>` : '') +
      `<div class="vk-scroll"><table class="vk-tree">${th}<tbody>${body}</tbody></table></div>` +
      notes.map(t => `<div class="vk-warn">${t}</div>`).join('');
  }

  function click(e, st) {
    const c = e.target.closest('[data-vk-set] .vk-chip');
    if (c) return { set: { [c.parentElement.dataset.vkSet]: c.dataset.v } };
    if (e.target.closest('[data-vk-unfold]')) return { set: { shut: [] } };
    const t = e.target.closest('[data-vk-toggle]');
    if (t) { const id = t.dataset.vkToggle, s = new Set(st.shut || []); s.has(id) ? s.delete(id) : s.add(id); return { set: { shut: [...s] } } }
    const l = e.target.closest('[data-vk-link]');
    if (l) return { link: { kind: l.dataset.vkLink, reading: l.dataset.reading || '', night: l.dataset.night || '' } };
    if (e.target.closest('.vk-x')) return null;
    const r = e.target.closest('[data-vk-row]');
    if (r) return { set: { open: st.open === r.dataset.vkRow ? '' : r.dataset.vkRow } };
    return null;
  }

  root.VizKit = { esc, fin, fmt, sgn, minus, chips, swatch, fold, dcell, colour, click, tree: { html: treeHTML, defaults: TREE } };
})(typeof window !== 'undefined' ? window : this);
