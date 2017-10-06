package main

// #cgo LDFLAGS: -L/Users/tomaspinho/projects/marian/build -lamunmt
// #include "stdlib.h"
// #include "./amunmt.h"
import "C"
import "unsafe"
import "time"
import "fmt"

func initAmun(options string) {
	optionsc := C.CString(options)
	C.init(optionsc)
	C.free(unsafe.Pointer(optionsc))
}

func translate(in []string) []string {
	var iLines []*C.char

	// Transform input strings to C strings
	for _, l := range in {
		iLine := C.CString(l)
		defer C.free(unsafe.Pointer(iLine))
		iLines = append(iLines, iLine)
	}
	iLines = append(iLines, nil)

	output := C.translate(&iLines[0])
	defer C.free(unsafe.Pointer(output)) // Free output array
	var oLines []string

	length := len(in) // exploiting the fact that length of input == length of output
	slice := (*[1 << 30]*C.char)(unsafe.Pointer(output))[0:length]

	// Transform output strings to C strings
	for _, ol := range slice {
		oLines = append(oLines, C.GoString(ol))
		C.free(unsafe.Pointer(ol)) // Free output line
	}

	return oLines
}

func main() {
	initAmun("-c models/amunmt.config")

	then := time.Now()
	output := translate([]string{"Hello words!"})

	fmt.Printf("Took %fs\n", time.Now().Sub(then).Seconds())
	println(output[0])
}
