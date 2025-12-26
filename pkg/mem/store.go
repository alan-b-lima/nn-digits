package mem

import (
	"encoding/base64"
	"encoding/binary"
	"errors"
	"math"
)

type Float64Slice []float64

func (m Float64Slice) MarshalJSON() ([]byte, error) {
	var bytes []byte
	for _, n := range m {
		bytes = binary.LittleEndian.AppendUint64(bytes, math.Float64bits(n))
	}

	buf := base64.StdEncoding.AppendEncode([]byte{'"'}, bytes)
	buf = append(buf, '"')

	return buf, nil
}

func (m *Float64Slice) UnmarshalJSON(buf []byte) error {
	if len(buf) < 2 || buf[0] != '"' || buf[len(buf)-1] != '"' {
		return errors.New("matrix must be a well-formed JSON string")
	}

	bytes, err := base64.StdEncoding.AppendDecode(nil, buf[1:len(buf)-1])
	if err != nil {
		return err
	}

	*m = make(Float64Slice, 0, len(bytes)/8)
	for len(bytes) > 0 {
		n := binary.LittleEndian.Uint64(bytes)
		*m = append(*m, math.Float64frombits(n))
		bytes = bytes[8:]
	}

	return nil
}
