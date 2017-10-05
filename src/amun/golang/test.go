package main

// #cgo LDFLAGS: -L${SRCDIR} -lamunmt
// #include "./amunmt.h"
import "C"

func main() {
	C.init(C.CString("test"))
}
