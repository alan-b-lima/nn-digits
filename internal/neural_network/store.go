package nn

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"unsafe"
)

func (nn NeuralNetwork) MarshalJSON() ([]byte, error) {
	return json.Marshal(*(*neural_network)(unsafe.Pointer(&nn)))
}

func (nn *NeuralNetwork) UnmarshalJSON(buf []byte) error {
	var vn neural_network
	if err := json.Unmarshal(buf, &vn); err != nil {
		return err
	}

	*nn = *(*NeuralNetwork)(unsafe.Pointer(&vn))
	return nil
}

type neural_network struct {
	Layers []layer `json:"layers"`
}

type layer struct {
	Weights        matrix `json:"weights"`
	Biases         matrix `json:"biases"`
	Activation     matrix `json:"activation"`
	WeightGradient matrix `json:"weight_gradient"`
	BiasGradient   matrix `json:"bias_gradient"`
}

type matrix struct {
	Rows int  `json:"rows"`
	Cols int  `json:"cols"`
	Data nums `json:"data"`
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
