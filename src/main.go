package main

import (
	"fmt"

	net "./nn/net"
)

func main() {
	fmt.Println("Running")
	net.Create([]int{2, 12, 2})
}
