//go:build wasm && js

package main

import (
	"encoding/json"
	"syscall/js"

	"github.com/alan-b-lima/nn-digits/internal/digits"
	"github.com/alan-b-lima/nn-digits/internal/neural_network"
)

func main() {
	var nn nn.NeuralNetwork

	js.Global().Set("load", js.FuncOf(Load(&nn)))
	js.Global().Set("classify", js.FuncOf(Classify(digits.NewClassifier(&nn))))

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

func Classify(classifier digits.Classifier) func(js.Value, []js.Value) any {
	return func(_ js.Value, args []js.Value) any {
		var arg js.Value
		if arg = args[0]; arg.Type() != js.TypeObject || !arg.InstanceOf(Array) {
			return nil
		}

		if arg.Get("length").Int() != len(digits.Request{}) {
			return nil
		}

		var data digits.Request
		for i := range len(data) {
			val := arg.Index(i)
			if val.Type() != js.TypeNumber {
				return nil
			}

			data[i] = val.Float()
		}

		res, err := classifier.Classify(data)
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
