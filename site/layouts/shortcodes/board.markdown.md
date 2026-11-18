{{- $b := site.Data.submissions.board -}}
{{- if and $b $b.rows }}
| # | model | overall |{{ range $b.columns }} {{ . }} |{{ end }}
|--:|---|--:|{{ range $b.columns }}--:|{{ end }}
{{- range $b.rows }}
| {{ .n }} | {{ .name }}{{ if eq .class "calibration" }}*{{ end }} | {{ printf "%.4f" .overall }} |{{ range .cells }} {{ if eq (printf "%v" .scaled) "<nil>" }}—{{ else }}{{ printf "%.3f" .scaled }}{{ end }} |{{ end }}
{{- end }}

\* a calibration run: an entry sent purely to test a new method and its performance on the official board. Each score is the scaled score the board ranks on.
{{ end -}}
