{{- with .Page.Resources.GetMatch (.Get "src") }}![{{ with $.Get "caption" }}{{ . }}{{ else }}figure{{ end }}]({{ .Permalink }}){{ end -}}
