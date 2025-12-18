//go:build wasm && js

package main

import (
	"encoding/json"
	"syscall/js"

	"github.com/alan-b-lima/nn-digits/internal/neural_network"
	"github.com/alan-b-lima/nn-digits/internal/service"
)

func main() {
	var nn nn.NeuralNetwork

	js.Global().Set("load", js.FuncOf(Load(&nn)))
	js.Global().Set("classify", js.FuncOf(Classify(service.NewClassifier(&nn))))
	select {}
}

var Array = js.Global().Get("Array")

func Load(nn *nn.NeuralNetwork) func(js.Value, []js.Value) any {
	return func(_ js.Value, args []js.Value) any {
		var arg js.Value

		if arg = args[0]; arg.Type() != js.TypeString {
			return nil
		}
		j := arg.String()

		err := json.Unmarshal([]byte(j), nn)
		return err == nil
	}
}

func Classify(id service.Classifier) func(js.Value, []js.Value) any {
	return func(_ js.Value, args []js.Value) any {
		var arg js.Value

		if arg = args[1]; arg.Type() != js.TypeNumber {
			return nil
		}
		width := arg.Int()

		if arg = args[2]; arg.Type() != js.TypeNumber {
			return nil
		}
		height := arg.Int()

		if arg = args[0]; arg.Type() != js.TypeObject || !arg.InstanceOf(Array) {
			return nil
		}
		length := arg.Get("length").Int()
		array := make([]float64, length)

		for i := range length {
			val := arg.Index(i)
			if val.Type() != js.TypeNumber {
				return nil
			}

			array[i] = val.Float()
		}

		res, err := id.Classify(service.Request{
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
