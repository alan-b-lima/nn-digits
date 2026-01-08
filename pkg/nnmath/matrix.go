package nnmath

import "fmt"

const safe = true

// Matrix is a matrix of float64s.
//
// Once initialized, the dimensions of the matrix cannot be changed. The
// underlying data slice, accessible with [Matrix.Data], is stable, i.e., it
// will never change, and can be modified as pleased, though appends may have
// unintended side effects.
// 
// For content placement, see [Matrix.Data].
type Matrix struct {
	rows int
	cols int
	data []float64
}

// Vector is an alias for [Matrix], it hints that the type is a column vector,
// but its treated no different from a matrix.
type Vector = Matrix

// MakeMat makes a new matrix with the given dimensions.
func MakeMat(rows, cols int) Matrix {
	return Matrix{
		rows: rows,
		cols: cols,
		data: make([]float64, rows*cols),
	}
}

// MakeMatData makes a new matrix with the given dimensions and uses the data
// slice as the underlying slice for the matrix.
//
// MakeMatData panics if len(data) is different from rows * cols.
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

// MakeVec makes a new column vector with the given height. Equivalent to
// [MakeMat](size, 1).
func MakeVec(size int) Vector {
	return MakeMat(size, 1)
}

// MakeVecData makes a new column vector with the given height and uses the
// data slice as the underlying slice for the matrix.
//
// MakeVecData panics if len(data) is different from size.
func MakeVecData(size int, data []float64) Vector {
	if safe {
		if len(data) != size {
			panic(fmt.Sprintf("data length %d does not match vector size %d", len(data), size))
		}
	}

	return MakeMatData(size, 1, data)
}

// Size returns the size, i.e., the height times the width.
func (M Matrix) Size() int {
	return M.rows * M.cols
}

// Dim returns the dimensions of the matrix.
func (M Matrix) Dim() (row, col int) {
	return M.rows, M.cols
}

// Rows returns the number of rows.
func (M Matrix) Rows() int {
	return M.rows
}

// Cols returns the number of columns.
func (M Matrix) Cols() int {
	return M.cols
}

// Data returns the underlying slice of the matrix.
//
// Data is guaranteed the always return the same slice for any matrix,
// modifiying it will modify the contents matrix itself.
//
// The data at (i, j) from M can be indexed as data[i*M.Cols() + j].
func (M Matrix) Data() []float64 {
	return M.data
}

// At returns the cell content at (row, col).
//
// At panics if (row, col) is out of range.
func (M Matrix) At(row, col int) float64 {
	if safe {
		if row < 0 || M.rows <= row || col < 0 || M.cols <= col {
			panic(fmt.Sprintf("index out of range [%d][%d] with length %d,%d", row, col, M.rows, M.cols))
		}
	}

	return M.data[row*M.cols+col]
}

// Set sets the cell content at (row, col).
//
// Set panics if (row, col) is out of range.
func (M Matrix) Set(row, col int, value float64) {
	if safe {
		if row < 0 || M.rows <= row || col < 0 || M.cols <= col {
			panic(fmt.Sprintf("index out of range [%d][%d] with length %d,%d", row, col, M.rows, M.cols))
		}
	}

	M.data[row*M.cols+col] = value
}

// Zero zeros out the entire matrix.
func Zero(A Matrix) {
	for i := range len(A.data) {
		A.data[i] = 0
	}
}

// Assign copies the contents from one matrix to another.
//
// Assign panics if the dimensions of the two matrices don't have the same
// dimensions.
func Assign(A Matrix, B Matrix) {
	if safe {
		if A.rows != B.rows || A.cols != B.cols {
			panic("matrix dimensions do not match")
		}
	}

	copy(A.data, B.data)
}

// Add adds two matrices. For R, A, B in [n x m], Add(R, A, B) describes
// R = A + B.
//
// Add panics if the dimensions of the three matrices don't match.
func Add(R Matrix, A, B Matrix) {
	if safe {
		if R.rows != A.rows || R.cols != A.cols || A.rows != B.rows || A.cols != B.cols {
			panic("matrix dimensions do not match")
		}
	}

	for i := range len(A.data) {
		R.data[i] = A.data[i] + B.data[i]
	}
}

// AddMul computes the multiplication of two matrices and adds the result to a
// third matrix. For R, A in [n x m], B in [n x p], C in [p x m],
// AddMul(R, A, B, C) describes R = A + B * C.
//
// AddMul panics if the dimensions of the three matrices don't match.
func AddMul(R Matrix, A, B, C Matrix) {
	if safe {
		if R.rows != A.rows || R.cols != A.cols || A.rows != B.rows || A.cols != C.cols || B.cols != C.rows {
			panic("matrix dimensions do not match")
		}
	}

	var i int
	for j := range B.rows {
		for k := range C.cols {
			sum := A.At(j, k)
			for l := range B.cols {
				sum += B.At(j, l) * C.At(l, k)
			}
			R.data[i] = sum
			i++
		}
	}
}

// AddSMul computes the scalar multiplication of a matrix and adds the result
// to a second matrix. For R, A, B in [n x m], AddSMul(R, A, s, B) describes
// R = A + s * B.
//
// AddSMul panics if the dimensions of the three matrices don't match.
func AddSMul(R Matrix, A Matrix, s float64, B Matrix) {
	if safe {
		if R.rows != A.rows || R.cols != A.cols || A.rows != B.rows || A.cols != B.cols {
			panic("matrix dimensions do not match")
		}
	}

	for i := range len(A.data) {
		R.data[i] = A.data[i] + s*B.data[i]
	}
}

// Mul multiplies two matrices. For R in [n x m], A in [n x p], B in [p x m],
// Mul(R, A, B) describes R = A * B.
//
// Mul panics if the dimensions of the three matrices don't match.
func Mul(R Matrix, A, B Matrix) {
	if safe {
		if R.rows != A.rows || R.cols != B.cols || A.cols != B.rows {
			panic("matrix dimensions do not match")
		}
	}

	var i int
	for j := range A.rows {
		for k := range B.cols {
			var sum float64
			for l := range A.cols {
				sum += A.At(j, l) * B.At(l, k)
			}
			R.data[i] = sum
			i++
		}
	}
}

// HMul computes the Hadamard product (element-wise multiplication) of two
// matrices. For R, A, B in [n x m], HMul(R, A, B) describes R = A ⊙ B.
//
// HMul panics if the dimensions of the three matrices don't match.
func HMul(R Matrix, A, B Matrix) {
	if safe {
		if R.rows != A.rows || R.cols != A.cols || A.rows != B.rows || A.cols != B.cols {
			panic("matrix dimensions do not match")
		}
	}

	for i := range len(A.data) {
		R.data[i] = A.data[i] * B.data[i]
	}
}

// SMul computes the scalar multiplication of a matrix. For R, A in [n x m],
// SMul(R, s, A) describes R = s * A.
//
// SMul panics if the dimensions of the two matrices don't match.
func SMul(R Matrix, s float64, A Matrix) {
	if safe {
		if R.rows != A.rows || R.cols != A.cols {
			panic("matrix dimensions do not match")
		}
	}

	for i := range len(A.data) {
		R.data[i] = s * A.data[i]
	}
}

// Dot computes the dot product of two matrices. For A, B in [n x m], Dot(A, B)
// describes A^T * B.
//
// Dot panics if the dimensions of the two matrices don't match, but not if
// such are not actually vector.
func Dot(A, B Matrix) float64 {
	if safe {
		if A.rows != B.rows || A.cols != B.cols {
			panic("matrix dimensions do not match")
		}
	}

	var sum float64
	for i := range len(A.data) {
		sum += A.data[i] * B.data[i]
	}

	return sum
}

// Apply applies a function to each element of a matrix. For R, A in [n x m],
// Apply(R, A, fn) describes R[i][j] = fn(A[i][j]).
//
// Apply panics if the dimensions of the two matrices don't match.
func Apply(R Matrix, A Matrix, fn func(float64) float64) {
	if safe {
		if R.rows != A.rows || R.cols != A.cols {
			panic("matrix dimensions do not match")
		}
	}

	for i := range len(A.data) {
		R.data[i] = fn(A.data[i])
	}
}
