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
    if (c) { const k = c.parentElement.dataset.vkSet;
      return { set: k.startsWith('f:') ? { f: Object.assign({}, st.f, { [k.slice(2)]: c.dataset.v }) } : { [k]: c.dataset.v } } }
    const so = e.target.closest('[data-vk-sort]');
    if (so) { const k = so.dataset.vkSort; return { set: st.sort === k ? { desc: st.desc ? '' : '1' } : { sort: k, desc: '1' } } }
    if (e.target.closest('[data-vk-q]')) return null;
    if (e.target.closest('[data-vk-unfold]')) return { set: { shut: [] } };
    const t = e.target.closest('[data-vk-toggle]');
    if (t) { const id = t.dataset.vkToggle, s = new Set(st.shut || []); s.has(id) ? s.delete(id) : s.add(id); return { set: { shut: [...s] } } }
    const l = e.target.closest('[data-vk-link]');
    if (l) return { link: { kind: l.dataset.vkLink, reading: l.dataset.reading || '', night: l.dataset.night || '', path: l.dataset.path || '' } };
    if (e.target.closest('.vk-x')) return null;
    const r = e.target.closest('[data-vk-row]');
    if (r) return { set: { open: st.open === r.dataset.vkRow ? '' : r.dataset.vkRow } };
    return null;
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

  /* ── list: rows of one kind, filtered by chips and a search box, sorted by a header, a row opening a card below it ──
     data: {columns: [{key, label, kind: gene|tag|text|num|pct, digits, title, mark}], rows: [{id, sub, ...}],
     filters: [{key, label, options: [{v, t}]}], search: placeholder, how}; state: {q, f: {key: v}, sort, desc, open}. */
  const LIST = { q: '', f: {}, sort: '', desc: '', open: '' };
  const cellOf = (c, r) => {
    const v = r[c.key];
    if (c.kind === 'gene') return `<td class="vk-g1"><b>${esc(v)}</b>${r.sub ? `<small>${esc(r.sub)}</small>` : ''}</td>`;
    if (c.kind === 'tag' || c.kind === 'text') return `<td class="vk-tag">${v == null || v === '' ? '<span class="vk-na">–</span>' : esc(v)}</td>`;
    if (c.kind === 'pct') return `<td class="vk-num">${fin(v) ? Math.round(100 * v) + ' %' : '<span class="vk-na">–</span>'}</td>`;
    return `<td class="vk-num">${fin(v) ? minus(fmt(v, c.digits)) : '<span class="vk-na">–</span>'}${c.mark && r[c.mark] ? ' <span class="vk-mk">●</span>' : ''}</td>`;
  };
  const rowText = (D, r) => [r.id, r.sub, ...(D.columns || []).filter(c => c.kind === 'tag' || c.kind === 'text').map(c => r[c.key])].join(' ').toLowerCase();
  const matches = (q, t) => { const w = String(q || '').toLowerCase().split(/\s+/).filter(Boolean); return w.every(x => t.includes(x)) };
  function listHTML(D, st) {
    const cols = D.columns || [], f = st.f || {};
    let rows = (D.rows || []).filter(r => Object.entries(f).every(([k, v]) => v === '' || v == null || String(r[k]) === String(v)));
    const col = cols.find(c => c.key === st.sort);
    if (col) {
      const s = st.desc ? -1 : 1, num = col.kind === 'num' || col.kind === 'pct';
      rows = rows.slice().sort((a, b) => { const x = a[col.key], y = b[col.key];
        if (x == null || x === '') return 1; if (y == null || y === '') return -1;
        return (num ? x - y : String(x).localeCompare(String(y))) * s });
    }
    const n = cols.length;
    const ctl = `<div class="vk-ctl">` + (D.filters || []).map(F => chips('f:' + F.key, F.label, F.options, f[F.key] ?? '')).join('') +
      `<span class="vk-set"><span class="vk-mi">find</span><input type="search" class="vk-q" data-vk-q="1" value="${esc(st.q || '')}" placeholder="${esc(D.search || '')}" spellcheck="false" autocomplete="off"></span>` +
      `<span class="vk-mi vk-cnt" data-vk-count="1"></span></div>`;
    const th = `<thead><tr>${cols.map(c => `<th class="${c.kind === 'num' || c.kind === 'pct' ? 'r' : ''}${st.sort === c.key ? ' on' : ''}" data-vk-sort="${esc(c.key)}" title="${esc(c.title || 'sort')}">${esc(c.label)}${st.sort === c.key ? (st.desc ? ' ↓' : ' ↑') : ''}</th>`).join('')}</tr></thead>`;
    const body = rows.map(r => {
      const on = st.open === r.id, hide = !matches(st.q, rowText(D, r));
      return `<tr class="vk-r${on ? ' vk-on' : ''}" data-vk-row="${esc(r.id)}" data-vk-text="${esc(rowText(D, r))}"${hide ? ' hidden' : ''}>${cols.map(c => cellOf(c, r)).join('')}</tr>` +
        (on ? `<tr class="vk-x vk-cardrow"><td colspan="${n}"><div class="vk-card" data-vk-card="${esc(r.id)}"><div class="vk-mi">loading…</div></div></td></tr>` : '');
    }).join('');
    return ctl + (D.how ? fold('how to read it', `<p class="vk-lgd">${esc(D.how)}</p>`) : '') +
      `<div class="vk-scroll vk-listwrap"><table class="vk-tree vk-list">${th}<tbody>${body}</tbody></table></div>`;
  }
  // the search box narrows the rows in place, so typing never redraws the view
  function listFilter(el, q) {
    let k = 0, n = 0;
    el.querySelectorAll('tr[data-vk-text]').forEach(tr => { n++; const ok = matches(q, tr.dataset.vkText); tr.hidden = !ok; if (ok) k++;
      const nx = tr.nextElementSibling; if (nx && nx.classList.contains('vk-cardrow')) nx.hidden = !ok });
    const c = el.querySelector('[data-vk-count]'); if (c) c.textContent = `${k} of ${n}`;
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
      else if (b.kind === 'table') inner = (b.how ? `<p class="vk-lgd">${esc(b.how)}</p>` : '') + `<table class="vk-tree vk-mini"><thead><tr>${(b.columns || []).map(c => `<th class="${c.kind === 'num' || c.kind === 'pct' ? 'r' : ''}">${esc(c.label)}</th>`).join('')}</tr></thead><tbody>${(b.rows || []).map(r => `<tr>${(b.columns || []).map(c => cellOf(c, r)).join('')}</tr>`).join('') || `<tr><td class="vk-na" colspan="${(b.columns || []).length}">none</td></tr>`}</tbody></table>`;
      return `<details class="vk-fold vk-blk"${b.open ? ' open' : ''}><summary>${esc(b.h)}</summary><div>${inner}</div></details>`;
    };
    return `<div class="vk-chead"><h3>${esc(C.title)}${C.name ? ` <span>${esc(C.name)}</span>` : ''}</h3>${(C.tags || []).filter(Boolean).map(t => `<span class="vk-b">${esc(t)}</span>`).join('')}` +
      `${C.sub ? `<div class="vk-mi">${esc(C.sub)}</div>` : ''}${C.path ? `<button type="button" class="vk-link" data-vk-link="path" data-path="${esc(C.path)}">${esc(C.path)}</button>` : ''}</div>` +
      `<div class="vk-blocks">${(C.blocks || []).map(block).join('')}</div>`;
  }

  root.VizKit = { esc, fin, fmt, sgn, minus, chips, swatch, fold, dcell, colour, click, md,
    tree: { html: treeHTML, defaults: TREE }, list: { html: listHTML, filter: listFilter, defaults: LIST }, card: { html: cardHTML } };
})(typeof window !== 'undefined' ? window : this);
