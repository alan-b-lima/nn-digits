package mem

func Take[E any](s *[]E, len int) []E {
	res := (*s)[:len]
	*s = (*s)[len:]
	return res
}
