package nn

import (
	"encoding/json"

	"github.com/alan-b-lima/nn-digits/pkg/mem"
)

func (nn NeuralNetwork) MarshalJSON() ([]byte, error) {
	if len(nn.Layers) == 0 {
		return []byte("{}"), nil
	}

	jn := neural_network{
		Dimensions: make([]int, 0, len(nn.Layers)+1),
		Layers:     nn.buf,
	}

	jn.Dimensions = append(jn.Dimensions, nn.Layers[0].Weights.Cols())
	for _, layer := range nn.Layers {
		jn.Dimensions = append(jn.Dimensions, layer.Weights.Rows())
	}

	return json.Marshal(jn)
}

func (nn *NeuralNetwork) UnmarshalJSON(buf []byte) error {
	var jn neural_network
	if err := json.Unmarshal(buf, &jn); err != nil {
		return err
	}

	*nn = NeuralNetwork{buf: jn.Layers}
	nn.Layers = slice_nn(nn.buf, jn.Dimensions...)

	return nil
}

type neural_network struct {
	Dimensions []int            `json:"dimensions"`
	Layers     mem.Float64Slice `json:"layers"`
}
