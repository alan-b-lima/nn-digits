//go:build wasm && js

package main

import (
	"encoding/json"
	"fmt"
	"syscall/js"

	"github.com/alan-b-lima/nn-digits/internal/digits"
	nn "github.com/alan-b-lima/nn-digits/internal/neural_network"
)

func main() {
	var nn nn.NeuralNetwork

	js.Global().Set("load", js.FuncOf(Load(&nn)))
	js.Global().Set("classify", js.FuncOf(Classify(digits.NewClassifier(&nn))))

	select {}
}

var (
	Array = js.Global().Get("Array")

	Error     = js.Global().Get("Error")
	TypeError = js.Global().Get("TypeError")
)

func Load(nn *nn.NeuralNetwork) func(js.Value, []js.Value) any {
	return func(_ js.Value, args []js.Value) any {
		if len(args) < 1 {
			return Error.New("expected 1 argument")
		}

		var arg js.Value

		if arg = args[0]; arg.Type() != js.TypeString {
			return TypeError.New("data is not a string")
		}
		j := arg.String()

		err := json.Unmarshal([]byte(j), nn)
		if err != nil {
			return Error.New(err.Error())
		}

		return nil
	}
}

func Classify(classifier digits.Classifier) func(js.Value, []js.Value) any {
	return func(_ js.Value, args []js.Value) any {
		if len(args) < 1 {
			return Error.New("expected 1 argument")
		}

		var arg js.Value
		if arg = args[0]; arg.Type() != js.TypeObject || !arg.InstanceOf(Array) {
			return TypeError.New("data is not an array")
		}

		if len := len(digits.Request{}); arg.Get("length").Int() != len {
			return Error.New(fmt.Sprintf("data is not of length %d", len))
		}

		var data digits.Request
		for i := range len(data) {
			val := arg.Index(i)
			if val.Type() != js.TypeNumber {
				return TypeError.New(fmt.Sprintf("data[%d] is not an number", i))
			}

			data[i] = val.Float()
		}

		res, err := classifier.Classify(&data)
		if err != nil {
			return Error.New(err.Error())
		}

		result := Array.New(len(res))
		for i, val := range res {
			result.SetIndex(i, val)
		}

		return result
	}
}
