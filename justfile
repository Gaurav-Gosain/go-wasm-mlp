compile-dev:
    GOOS=js GOARCH=wasm go build -v -o assets/tiny.wasm cmd/wasm/main.go

compile-prod:
    GOOS=wasi GOARCH=wasm tinygo build -o assets/tiny.wasm cmd/wasm/main.go

setup-dev:
    cp $(go env GOROOT)/misc/wasm/wasm_exec.js assets/

setup-prod:
    cp $(tinygo env TINYGOROOT)/targets/wasm_exec.js assets/

serve-dev port=":8080" root="assets": setup-dev
    go run cmd/server/main.go -port {{port}} -root {{root}}

serve-prod port=":8080" root="assets": setup-prod
    go run cmd/server/main.go -port {{port}} -root {{root}}

dev port=":8080" root="assets": compile-dev
    just serve-dev {{port}} {{root}}

prod port=":8080" root="assets": compile-prod
    just serve-prod {{port}} {{root}}

