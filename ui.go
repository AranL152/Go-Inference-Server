package main

import (
	"embed"
	"io/fs"
	"net/http"
)

// built by cd ui && pnpm publish
//
//go:embed all:webui
var webUI embed.FS

func uiHandler() http.Handler {
	sub, _ := fs.Sub(webUI, "webui")
	return http.FileServer(http.FS(sub))
}
