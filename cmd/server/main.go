package main

import (
	"flag"
	"log"
	"net/http"
)

var (
	addr = flag.String("port", ":8080", "http service address")
	root = flag.String("root", "assets", "path to serve")
)

func main() {
	flag.Parse()
	log.Printf("Serving %q at %s", *root, *addr)
	err := http.ListenAndServe(*addr, http.FileServer(http.Dir(*root)))
	if err != nil {
		log.Fatal(err)
	}
}
