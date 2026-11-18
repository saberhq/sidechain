{{- /* A page as Markdown (index.md beside index.html). */ -}}
# {{ .Title }}

> {{ or .Description site.Params.description }}

By {{ site.Params.author }}{{ if not .Date.IsZero }} · {{ .Date.Format "2006-01-02" }}{{ end }}{{ if and (not .Lastmod.IsZero) (ne (.Lastmod.Format "2006-01-02") (.Date.Format "2006-01-02")) }} · updated {{ .Lastmod.Format "2006-01-02" }}{{ end }} · web page: {{ .Permalink }}
{{- if eq .Layout "controls" }}

The figures are interactive on the web page; the data behind them: {{ "data/controls/index.json" | absURL }}
{{- end }}

{{ replaceRE `\]\(/` (printf "](%s" site.BaseURL) .RenderShortcodes }}
