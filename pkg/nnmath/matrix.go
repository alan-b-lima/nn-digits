package nnmath

import (
	"fmt"
	"math/rand"
)

type Matrix struct {
	Rows int
	Cols int
	Data []float64
}

type Vector = Matrix

func MakeMat(rows, cols int) Matrix {
	return Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([]float64, rows*cols),
	}
}

func MakeMatData(rows, cols int, data []float64) Matrix {
	if len(data) != rows*cols {
		panic(fmt.Sprintf("data length %d does not match matrix size %d,%d", len(data), rows, cols))
	}

	return Matrix{
		Rows: rows,
		Cols: cols,
		Data: data,
	}
}

func MakeMatRandom(rows, cols int) Matrix {
	M := MakeMat(rows, cols)

	for i := range rows * cols {
		M.Data[i] = rand.NormFloat64()
	}

	return M
}

func MakeVec(size int) Vector {
	return MakeMat(size, 1)
}

func MakeVecData(size int, data []float64) Vector {
	if len(data) != size {
		panic(fmt.Sprintf("data length %d does not match vector size %d", len(data), size))
	}

	return MakeMatData(size, 1, data)
}

func MakeVecRandom(size int) Vector {
	V := MakeVec(size)

	for i := range size {
		V.Data[i] = rand.NormFloat64()
	}

	return V
}

func (M Matrix) Dim() (row, col int) {
	return M.Rows, M.Cols
}

func (M Matrix) At(row, col int) float64 {
	if row < 0 || M.Rows <= row || col < 0 || M.Cols <= col {
		panic(fmt.Sprintf("index out of range [%d][%d] with length %d,%d", row, col, M.Rows, M.Cols))
	}

	return M.Data[row*M.Cols+col]
}

func (M Matrix) Set(row, col int, value float64) {
	if row < 0 || M.Rows <= row || col < 0 || M.Cols <= col {
		panic(fmt.Sprintf("index out of range [%d][%d] with length %d,%d", row, col, M.Rows, M.Cols))
	}

	M.Data[row*M.Cols+col] = value
}

func Add(A, B Matrix) Matrix {
	if A.Rows != B.Rows || A.Cols != B.Cols {
		panic("matrix dimensions do not match")
	}

	C := MakeMat(A.Rows, A.Cols)

	for i := range A.Rows {
		for j := range A.Cols {
			C.Set(i, j, A.At(i, j)+B.At(i, j))
		}
	}

	return C
}

func Mul(A, B Matrix) Matrix {
	if A.Cols != B.Rows {
		panic("matrix dimensions do not match")
	}

	C := MakeMat(A.Rows, B.Cols)

	for i := range A.Rows {
		for j := range B.Cols {
			var sum float64
			for k := range A.Cols {
				sum += A.At(i, k) * B.At(k, j)
			}
			C.Set(i, j, sum)
		}
	}

	return C
}

func HMul(A, B Matrix) Matrix {
	if A.Rows != B.Rows || A.Cols != B.Cols {
		panic("matrix dimensions do not match")
	}

	C := MakeMat(A.Rows, A.Cols)

	for i := range A.Rows {
		for j := range A.Cols {
			C.Set(i, j, A.At(i, j)*B.At(i, j))
		}
	}

	return C
}

func SMul(s float64, A Matrix) Matrix {
	B := MakeMat(A.Rows, A.Cols)

	for i := range A.Rows {
		for j := range A.Cols {
			B.Set(i, j, s*A.At(i, j))
		}
	}

	return B
}

func Apply(A Matrix, fn func(float64) float64) Matrix {
	R := MakeMat(A.Rows, A.Cols)

	for i := range A.Rows {
		for j := range A.Cols {
			R.Set(i, j, fn(A.At(i, j)))
		}
	}

	return R
}
