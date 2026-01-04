package nnmath

import "fmt"

const safe = true

type Matrix struct {
	rows int
	cols int
	data []float64
}

type Vector = Matrix

func MakeMat(rows, cols int) Matrix {
	return Matrix{
		rows: rows,
		cols: cols,
		data: make([]float64, rows*cols),
	}
}

func MakeMatData(rows, cols int, data []float64) Matrix {
	if safe {
		if len(data) != rows*cols {
			panic(fmt.Sprintf("data length %d does not match matrix size %d,%d", len(data), rows, cols))
		}
	}

	return Matrix{
		rows: rows,
		cols: cols,
		data: data,
	}
}

func MakeVec(size int) Vector {
	return MakeMat(size, 1)
}

func MakeVecData(size int, data []float64) Vector {
	if safe {
		if len(data) != size {
			panic(fmt.Sprintf("data length %d does not match vector size %d", len(data), size))
		}
	}

	return MakeMatData(size, 1, data)
}

func (M Matrix) Size() int {
	return M.rows * M.cols
}

func (M Matrix) Dim() (row, col int) {
	return M.rows, M.cols
}

func (M Matrix) Rows() int {
	return M.rows
}

func (M Matrix) Cols() int {
	return M.cols
}

func (M Matrix) Data() []float64 {
	return M.data
}

func (M Matrix) At(row, col int) float64 {
	if safe {
		if row < 0 || M.rows <= row || col < 0 || M.cols <= col {
			panic(fmt.Sprintf("index out of range [%d][%d] with length %d,%d", row, col, M.rows, M.cols))
		}
	}

	return M.data[row*M.cols+col]
}

func (M Matrix) Set(row, col int, value float64) {
	if safe {
		if row < 0 || M.rows <= row || col < 0 || M.cols <= col {
			panic(fmt.Sprintf("index out of range [%d][%d] with length %d,%d", row, col, M.rows, M.cols))
		}
	}

	M.data[row*M.cols+col] = value
}

func Add(A, B Matrix) Matrix {
	C := MakeMat(A.rows, A.cols)
	AddP(C, A, B)
	return C
}

func Mul(A, B Matrix) Matrix {
	C := MakeMat(A.rows, B.cols)
	MulP(C, A, B)
	return C
}

func HMul(A, B Matrix) Matrix {
	C := MakeMat(A.rows, A.cols)
	HMulP(C, A, B)
	return C
}

func SMul(s float64, A Matrix) Matrix {
	B := MakeMat(A.rows, A.cols)
	SMulP(B, s, A)
	return B
}

func Apply(A Matrix, fn func(float64) float64) Matrix {
	R := MakeMat(A.rows, A.cols)
	ApplyP(R, A, fn)
	return R
}

func Zero(A Matrix) {
	for i := range len(A.data) {
		A.data[i] = 0
	}
}

func Assign(A Matrix, B Matrix) {
	if safe {
		if A.rows != B.rows || A.cols != B.cols {
			panic("matrix dimensions do not match")
		}
	}

	copy(A.data, B.data)
}

func AddP(R Matrix, A, B Matrix) {
	if safe {
		if A.rows != B.rows || A.cols != B.cols || R.rows != A.rows || R.cols != A.cols {
			panic("matrix dimensions do not match")
		}
	}

	for i := range len(A.data) {
		R.data[i] = A.data[i] + B.data[i]
	}
}

func AddMulP(R Matrix, A, B, C Matrix) {
	if safe {
		if B.cols != C.rows || A.rows != B.rows || A.cols != C.cols || R.rows != A.rows || R.cols != A.cols {
			panic("matrix dimensions do not match")
		}
	}

	for i := range B.rows {
		for j := range C.cols {
			sum := A.At(i, j)
			for k := range B.cols {
				sum += B.At(i, k) * C.At(k, j)
			}
			R.Set(i, j, sum)
		}
	}
}

func MulP(R Matrix, A, B Matrix) {
	if safe {
		if A.cols != B.rows || R.rows != A.rows || R.cols != B.cols {
			panic("matrix dimensions do not match")
		}
	}

	for i := range A.rows {
		for j := range B.cols {
			var sum float64
			for k := range A.cols {
				sum += A.At(i, k) * B.At(k, j)
			}
			R.Set(i, j, sum)
		}
	}
}

func HMulP(R Matrix, A, B Matrix) {
	if safe {
		if A.rows != B.rows || A.cols != B.cols || R.rows != A.rows || R.cols != A.cols {
			panic("matrix dimensions do not match")
		}
	}

	for i := range len(A.data) {
		R.data[i] = A.data[i] * B.data[i]
	}
}

func SMulP(R Matrix, s float64, A Matrix) {
	if safe {
		if R.rows != A.rows || R.cols != A.cols {
			panic("matrix dimensions do not match")
		}
	}

	for i := range len(A.data) {
		R.data[i] = s * A.data[i]
	}
}

func ApplyP(R Matrix, A Matrix, fn func(float64) float64) {
	if safe {
		if R.rows != A.rows || R.cols != A.cols {
			panic("matrix dimensions do not match")
		}
	}

	for i := range len(A.data) {
		R.data[i] = fn(A.data[i])
	}
}
