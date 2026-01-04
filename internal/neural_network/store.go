package nn

import (
	"encoding/json"

	"github.com/alan-b-lima/nn-digits/pkg/mem"
)

func (nn *NeuralNetwork) MarshalJSON() ([]byte, error) {
	if len(nn.layers) == 0 {
		return []byte("{}"), nil
	}

	nn.mu.RLock()
	defer nn.mu.RUnlock()

	jn := neural_network{
		Dimensions: nn.Dims(),
		Layers:     nn.buf,
	}

	return json.Marshal(jn)
}

func (nn *NeuralNetwork) UnmarshalJSON(buf []byte) error {
	var jn neural_network
	if err := json.Unmarshal(buf, &jn); err != nil {
		return err
	}

	nn.mu.Lock()
	defer nn.mu.Unlock()

	*nn = NeuralNetwork{buf: jn.Layers}

	nn.layers = slice_nn(nn.buf, jn.Dimensions...)
	nn.comp = mem.NewPool(nn.new_comp)
	nn.learn = mem.NewPool(nn.new_learn)

	return nil
}

type neural_network struct {
	Dimensions []int            `json:"dimensions"`
	Layers     mem.Float64Slice `json:"layers"`
}
