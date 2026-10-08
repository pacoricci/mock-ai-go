// Package docs serves the OpenAPI specification and its Swagger UI page.
package docs

import (
	_ "embed"
	"html/template"
	"net/http"
)

//go:embed openapi.yaml
var openAPISpec []byte

const swaggerUIPage = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>mock-ai-go API documentation</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = function() {
      window.ui = SwaggerUIBundle({
        url: "/openapi.yaml",
        dom_id: "#swagger-ui"
      });
    };
  </script>
</body>
</html>`

var swaggerUITemplate = template.Must(template.New("swagger-ui").Parse(swaggerUIPage))

// NewHandler constructs an HTTP handler for the API documentation routes.
func NewHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /docs", serveSwaggerUI)
	mux.HandleFunc("GET /docs/", serveSwaggerUI)
	mux.HandleFunc("GET /openapi.yaml", serveOpenAPISpec)
	return mux
}

func serveSwaggerUI(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := swaggerUITemplate.Execute(w, nil); err != nil {
		return
	}
}

func serveOpenAPISpec(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/yaml; charset=utf-8")
	_, _ = w.Write(openAPISpec)
}
