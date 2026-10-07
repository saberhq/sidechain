/* The landing page: the way back to the top, the two countdowns, and the plot's readout.
   Everything here is an enhancement. The figures, the plot and its links are drawn at build
   time (layouts/_partials/standing.html), the rows open with no script (they are <details>),
   and with scripts off a point is a plain link to its row. */
(function () {
  'use strict';

  var calm = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.querySelectorAll('[data-scroll-top]').forEach(function (btn) {
    btn.addEventListener('click', function () { window.scrollTo({ top: 0, behavior: calm ? 'auto' : 'smooth' }); });
  });

  /* Days to a date, counted in the visitor's own calendar days. A date already past keeps
     the build-time fallback, which prints the date itself. Once a count is showing, the line
     under it (hidden until then) gives the dates. */
  var MS_DAY = 86400000, counted = false;
  document.querySelectorAll('[data-days-to]').forEach(function (el) {
    var m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(el.getAttribute('data-days-to') || '');
    if (!m) return;
    var now = new Date();
    var days = Math.round((Date.UTC(+m[1], m[2] - 1, +m[3]) - Date.UTC(now.getFullYear(), now.getMonth(), now.getDate())) / MS_DAY);
    if (days < 0) return;
    el.textContent = String(days);
    var unit = document.createElement('small');
    unit.textContent = 'd';
    el.appendChild(unit);
    counted = true;
  });
  if (counted) document.querySelectorAll('[data-days-sub]').forEach(function (el) { el.hidden = false; });

  var fig = document.querySelector('[data-standing]');
  var plot = fig && fig.querySelector('[data-plot]');
  var axis = fig && fig.querySelector('[data-axis]');
  var readout = axis && axis.querySelector('[data-readout]');
  var card = fig && fig.querySelector('[data-card]');
  if (!plot || !readout || !card) return;
  var points = Array.prototype.slice.call(plot.querySelectorAll('a.sc-pt'));
  if (!points.length) return;
  /* The readout says what the browser's own tooltip would, so the tooltip goes. */
  points.forEach(function (pt) { pt.removeAttribute('title'); });

  function rowOf(pt) {
    var id = (pt.getAttribute('href') || '').slice(1);
    return id ? document.getElementById(id) : null;
  }

  function piece(tag, text, cls) {
    var el = document.createElement(tag);
    el.textContent = text;
    if (cls) el.className = cls;
    return el;
  }

  /* Two states. `hot` is the point under the pointer or the keyboard; `chosen` is the point
     that was clicked, and it holds the line and shows its model card until it is let go. */
  var hot = null, chosen = null;

  /* The readout takes the line under the plot, where the dates are, so it never covers a
     point: the entry's name, then its date, score, rank and share (a phone drops the date and
     the share, which are in the row). */
  function write(pt) {
    var d = pt.dataset;
    readout.textContent = '';
    readout.appendChild(piece('b', d.name));
    readout.appendChild(piece('span', ' · ' + d.date, 'sc-wide'));
    readout.appendChild(document.createTextNode(' · ' + d.overall + ' · ' + d.rank));
    readout.appendChild(piece('span', ' · ' + d.share, 'sc-wide'));
    if (pt === chosen) {
      var close = piece('button', '×', 'sc-spark-close');
      close.type = 'button';
      close.setAttribute('aria-label', 'Close ' + d.name);
      close.addEventListener('click', function () { choose(null); });
      readout.appendChild(close);
    }
    readout.hidden = false;
    axis.classList.add('is-reading');
  }

  function light(pt, on) {
    pt.classList.toggle('is-hot', on);
    var r = rowOf(pt);
    if (r) r.classList.toggle('is-hot', on);
  }

  function rest() {
    if (hot) { light(hot, false); hot = null; }
    if (chosen) { write(chosen); return; }
    readout.hidden = true;
    axis.classList.remove('is-reading');
  }

  function show(pt) {
    if (hot === pt) return;
    if (hot) light(hot, false);
    hot = pt;
    light(pt, true);
    write(pt);
  }

  /* Choosing a point puts its model card under the plot: the paragraph its row opens to,
     copied from that row so the two can never disagree. Choosing it again lets it go, and so
     do the close mark, Escape and a click anywhere else. */
  function choose(pt) {
    if (chosen) chosen.classList.remove('is-chosen');
    chosen = pt === chosen ? null : pt;
    card.textContent = '';
    card.hidden = true;
    if (chosen) {
      chosen.classList.add('is-chosen');
      var r = rowOf(chosen), body = r && r.querySelector('.sc-entry-body');
      var text = body && (body.querySelector('p') || body.querySelector('.sc-entry-meta'));
      if (text) { card.appendChild(text.cloneNode(true)); card.hidden = false; }
      write(chosen);
    } else {
      rest();
    }
  }

  /* The pointer only has to be closest to a point, not on it. */
  plot.addEventListener('pointermove', function (ev) {
    var best = null, bestD = Infinity;
    points.forEach(function (pt) {
      var b = pt.getBoundingClientRect();
      var dx = ev.clientX - (b.left + b.width / 2), dy = ev.clientY - (b.top + b.height / 2);
      var dist = dx * dx + dy * dy;
      if (dist < bestD) { bestD = dist; best = pt; }
    });
    if (best && bestD <= 48 * 48) show(best); else rest();
  });
  plot.addEventListener('pointerleave', rest);
  points.forEach(function (pt) {
    pt.addEventListener('focus', function () { show(pt); });
    pt.addEventListener('blur', rest);
    pt.addEventListener('click', function (ev) {
      ev.preventDefault();
      choose(pt);
    });
  });
  document.addEventListener('keydown', function (ev) {
    if (ev.key === 'Escape' && chosen) choose(null);
  });
  /* A click anywhere else lets the chosen point go. Its own card and readout do not count
     as elsewhere, so the card's text can still be selected. */
  document.addEventListener('click', function (ev) {
    if (!chosen) return;
    var t = ev.target;
    if ((t.closest && t.closest('a.sc-pt')) || card.contains(t) || readout.contains(t)) return;
    choose(null);
  });

  /* A row lights its point, as a point lights its row. */
  document.querySelectorAll('details.sc-entry[id]').forEach(function (row) {
    var pt = plot.querySelector('a.sc-pt[href="#' + row.id + '"]');
    if (!pt) return;
    var summary = row.querySelector('summary');
    summary.addEventListener('pointerenter', function () { pt.classList.add('is-hot'); });
    summary.addEventListener('pointerleave', function () { if (hot !== pt) pt.classList.remove('is-hot'); });
  });
})();
