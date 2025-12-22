package nn

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
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
	Dimensions []int `json:"dimensions"`
	Layers     nums  `json:"layers"`
}

type nums []float64

func (m nums) MarshalJSON() ([]byte, error) {
	var bytes []byte
	for _, n := range m {
		bytes = binary.LittleEndian.AppendUint64(bytes, math.Float64bits(n))
	}

	buf := base64.StdEncoding.AppendEncode([]byte{'"'}, bytes)
	buf = append(buf, '"')

	return buf, nil
}

func (m *nums) UnmarshalJSON(buf []byte) error {
	if len(buf) < 2 || buf[0] != '"' || buf[len(buf)-1] != '"' {
		return errors.New("matrix must be a well-formed JSON string")
	}

	bytes, err := base64.StdEncoding.AppendDecode(nil, buf[1:len(buf)-1])
	if err != nil {
		return err
	}

	*m = make(nums, 0, len(bytes)/8)
	for len(bytes) > 0 {
		n := binary.LittleEndian.Uint64(bytes)
		*m = append(*m, math.Float64frombits(n))
		bytes = bytes[8:]
	}

	return nil
}
