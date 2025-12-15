//go:build wasm && js

package main

import (
	"syscall/js"

	serve "github.com/alan-b-lima/nn-digits/internal/service"
	"github.com/alan-b-lima/nn-digits/internal/service/wasm"
)

func main() {
	serve := wasm.New()
	js.Global().Set("classify", js.FuncOf(Classify(serve)))
	
	select {}
}

var Array = js.Global().Get("Array")

func Classify(id serve.Classifier) func(js.Value, []js.Value) any {
	return func(_ js.Value, args []js.Value) any {
		var width, height int
		var array []float64

		var arg js.Value

		if arg = args[1]; arg.Type() != js.TypeNumber {
			return nil
		}
		width = arg.Int()

		if arg = args[2]; arg.Type() != js.TypeNumber {
			return nil
		}
		height = arg.Int()

		if arg = args[0]; arg.Type() != js.TypeObject || !arg.InstanceOf(Array) {
			return nil
		}
		length := arg.Get("length").Int()
		array = make([]float64, length)

		for i := range length {
			val := arg.Index(i)
			if val.Type() != js.TypeNumber {
				return nil
			}

			array[i] = val.Float()
		}

		res, err := id.Classify(serve.Request{
			Width:  width,
			Height: height,
			Data:   array,
		})
		if err != nil {
			return nil
		}

		result := Array.New(len(res))
		for i, val := range res {
			result.SetIndex(i, val)
		}

		return result
	}
}
