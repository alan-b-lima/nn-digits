package repl

import (
	"fmt"
	"math"
	"strings"
)

const (
	_UpRight    = '╰'
	_DownRight  = '╭'
	_Horizontal = '─'
	_Vertical   = '│'
	_UpLeft     = '╯'
	_DownLeft   = '╮'

	_BrailleBase = '\u2800'
)

func Graph(points []float64, width, height int) (int, string) {
	width, height = width-2, height-4
	domain, image := 2*width, 4*height

	if len(points) > domain {
		points = points[len(points)-domain:]
	}

	minp, maxp := math.Inf(1), math.Inf(-1)
	for _, p := range points {
		minp = min(minp, p)
		maxp = max(maxp, p)
	}

	mat := make([]int, domain)

	x := domain - len(points)
	for _, p := range points {
		y := int(math.Round(float64(image+1) * (p - minp) / (maxp - minp)))
		y = min(max(0, y), image-1)

		mat[x] = image - 1 - y
		x++
	}
	for i := range domain - len(points) {
		mat[i] = -1
	}

	var b strings.Builder
	b.Grow(3*(2*width+2*height+len(points)+4) + width*height - len(points) + width - 1)

	b.WriteRune(_DownRight)
	for range width {
		b.WriteRune(_Horizontal)
	}
	b.WriteRune(_DownLeft)

	for y := range height {
		b.WriteRune(_Vertical)
		for x := range width {
			braille := _BrailleBase

			if val := mat[x*2]; val >= 0 && y == val/4 {
				rem := val % 4
				if rem < 3 {
					braille |= rune(1 << rem)
				} else {
					braille |= rune(1 << 6)
				}
			}

			if val := mat[x*2+1]; val >= 0 && y == val/4 {
				rem := val % 4
				if rem < 3 {
					braille |= rune(1 << (rem + 3))
				} else {
					braille |= rune(1 << 7)
				}
			}

			if braille == _BrailleBase {
				b.WriteByte(' ')
			} else {
				b.WriteRune(braille)
			}
		}
		b.WriteRune(_Vertical)
	}

	b.WriteRune(_UpRight)
	for range width {
		b.WriteRune(_Horizontal)
	}
	b.WriteRune(_UpLeft)

	cc := fmt.Sprintf("max: %f, min: %f\n", maxp, minp)
	for range width + 2 - len(cc) {
		b.WriteByte(' ')
	}
	b.WriteString(cc)

	return len(points), b.String()
}
