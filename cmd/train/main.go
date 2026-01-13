package main

import (
	"os"

	"github.com/alan-b-lima/nn-digits/ui/repl"
)

func main() {
	repl.New(os.Stdout, os.Stdin)
}
