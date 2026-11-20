{{- /* The control-cells page's figures as text, one row per context, for the page's Markdown copy. Reads
       static/data/controls/ (index, tables, guides_<X>, genes_<X>), which scripts/site_controls.py writes. */ -}}
{{- $dir := "static/data/controls/" }}
{{- if and (os.FileExists (print $dir "index.json")) (os.FileExists (print $dir "tables.json")) }}
{{- $idx := os.ReadFile (print $dir "index.json") | transform.Unmarshal }}
{{- $tab := os.ReadFile (print $dir "tables.json") | transform.Unmarshal }}
{{- $n := 0 }}{{ range $idx.contexts }}{{ $n = .n_cells }}{{ end }}
{{- $cols := slice (dict "k" "total_umi" "h" "UMIs" "d" 0) (dict "k" "n_genes" "h" "genes" "d" 0)
      (dict "k" "pct_mt" "h" "mito UMI %" "d" 2) (dict "k" "pct_top50" "h" "top-50 UMI %" "d" 1)
      (dict "k" "complexity" "h" "complexity" "d" 3) (dict "k" "ifn_score" "h" "IFN score" "d" 2)
      (dict "k" "stress_score" "h" "stress score" "d" 2) }}

## The figures, in numbers

The web page draws its figures in the browser. These are the numbers they show, one row per context, each over its
{{ lang.FormatNumber 0 $n }} control cells. The maps (one dot per cell) and the violins are not reduced to numbers here;
their data is in the files linked above.

### 01 · Maps & QC

Medians per cell, and the share of cells called in each phase.

| context | round |{{ range $cols }} {{ .h }} |{{ end }} G1 · S · G2/M % |
|---|---|{{ range $cols }}--:|{{ end }}--:|
{{ range $c, $s := $idx.contexts -}}
| {{ $c }} | {{ $s.round }} |{{ range $col := $cols }} {{ if isset $s.median $col.k }}{{ lang.FormatNumber $col.d (index $s.median $col.k) }}{{ else }}–{{ end }} |{{ end }} {{ lang.FormatNumber 0 (mul 100 $s.phase_share.G1) }} · {{ lang.FormatNumber 0 (mul 100 $s.phase_share.S) }} · {{ lang.FormatNumber 0 (mul 100 $s.phase_share.G2M) }} |
{{ end }}
{{- with $tab.joint }}
{{- $lo := 1.0 }}
{{- range . }}{{ $top := 0.0 }}{{ range .by_context }}{{ if gt . $top }}{{ $top = . }}{{ end }}{{ end }}{{ $f := div (float $top) .cells }}{{ if lt $f $lo }}{{ $lo = $f }}{{ end }}{{ end }}
On the joint map of all six, one context holds at least {{ lang.FormatNumber 2 (mul 100 $lo) }} % of the cells in each of
its {{ len . }} clusters.
{{ end }}
The cluster table, summarised for each context's own map and for the same cells after depth and the two cell-cycle
scores are regressed out: how many clusters, the smallest one's cells, the fewest of the 46 guides any cluster holds,
the largest share of a cluster's cells that one guide carries (2.2 %, 1/46, if guides mix evenly) with that cluster's
size, and the cluster with the lowest median UMIs with its share of the context.

| context | map | clusters | smallest, cells | fewest guides | top guide, highest | lowest median UMIs · share of context |
|---|---|--:|--:|--:|--:|--:|
{{ range $c, $t := $tab.contexts -}}
{{ range $key := slice "cluster" "cluster_resid" -}}
{{ $rows := index $t $key -}}
{{ $small := 0.0 }}{{ $fewest := 0.0 }}{{ $top := dict }}{{ $low := dict -}}
{{ range $i, $r := $rows -}}
{{ if or (eq $i 0) (lt $r.cells $small) }}{{ $small = $r.cells }}{{ end -}}
{{ if or (eq $i 0) (lt $r.guides $fewest) }}{{ $fewest = $r.guides }}{{ end -}}
{{ if or (eq $i 0) (gt $r.top_guide $top.top_guide) }}{{ $top = $r }}{{ end -}}
{{ if or (eq $i 0) (lt $r.total_umi $low.total_umi) }}{{ $low = $r }}{{ end -}}
{{ end -}}
| {{ $c }} | {{ cond (eq $key "cluster") "own" "regressed" }} | {{ len $rows }} | {{ lang.FormatNumber 0 $small }} | {{ lang.FormatNumber 0 $fewest }} | {{ lang.FormatNumber 1 (mul 100 $top.top_guide) }} % ({{ lang.FormatNumber 0 $top.cells }} cells) | {{ lang.FormatNumber 0 $low.total_umi }} · {{ lang.FormatNumber 1 (mul 100 $low.share) }} % |
{{ end -}}
{{ end }}
### 02 · Guides & clones

Each guide's cells against 500 shuffles of the guide labels: how many of the 46 guides pass q < 0.05
(Benjamini–Hochberg) on the measured and on the regressed distance, and which; how many fall into the regressed
clusters in proportions unlike the context's (cluster skew, a chi-square test); the largest regressed distance z; and
the range of the guides' median UMIs.

| context | q < 0.05, measured | q < 0.05, regressed | which | cluster skew q < 0.05 | largest z, regressed | median UMIs per guide, lowest – highest |
|---|--:|--:|---|--:|--:|--:|
{{ range $c, $_ := $idx.contexts -}}
{{ $p := printf "%sguides_%s.json" $dir $c -}}
{{ if os.FileExists $p -}}
{{ $r := (os.ReadFile $p | transform.Unmarshal).rows -}}
{{ $m := 0 }}{{ $g := 0 }}{{ $k := 0 }}{{ $which := slice }}{{ $zmax := 0.0 }}{{ $lo := 0.0 }}{{ $hi := 0.0 -}}
{{ range $i, $name := $r.guide -}}
{{ $q := index $r.dist_q $i }}{{ $qr := index $r.dist_q_resid $i }}{{ $u := index $r.median_umi $i -}}
{{ if lt $q 0.05 }}{{ $m = add $m 1 }}{{ end }}{{ if lt $qr 0.05 }}{{ $g = add $g 1 }}{{ end -}}
{{ if or (lt $q 0.05) (lt $qr 0.05) }}{{ $which = $which | append $name }}{{ end -}}
{{ if lt (index $r.chi2_q_resid $i) 0.05 }}{{ $k = add $k 1 }}{{ end -}}
{{ if or (eq $i 0) (gt (index $r.dist_z_resid $i) $zmax) }}{{ $zmax = index $r.dist_z_resid $i }}{{ end -}}
{{ if or (eq $i 0) (lt $u $lo) }}{{ $lo = $u }}{{ end }}{{ if or (eq $i 0) (gt $u $hi) }}{{ $hi = $u }}{{ end -}}
{{ end -}}
| {{ $c }} | {{ $m }} | {{ $g }} | {{ with $which }}{{ delimit . ", " }}{{ else }}none{{ end }} | {{ $k }} | {{ lang.FormatNumber 2 $zmax }} | {{ lang.FormatNumber 0 $lo }} – {{ lang.FormatNumber 0 $hi }} |
{{ end -}}
{{ end }}
### 03 · Highest-expressed genes

The first five of the figure's 20 genes holding the most UMIs, each with its median share of a cell's UMIs (the
figure adds the 25–75 % and 5–95 % ranges).

| context | gene and median % of a cell's UMIs |
|---|---|
{{ range $c, $_ := $idx.contexts -}}
{{ $p := printf "%sgenes_%s.json" $dir $c -}}
{{ if os.FileExists $p -}}
| {{ $c }} | {{ range $i, $r := first 5 (os.ReadFile $p | transform.Unmarshal).top20_share }}{{ if $i }} · {{ end }}{{ $r.gene }} {{ lang.FormatNumber 2 (index $r.q 2) }}{{ end }} |
{{ end -}}
{{ end }}
{{- end }}
