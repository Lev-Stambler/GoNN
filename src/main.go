package main

import (
	"fmt"
	"math/rand"

	go_net "./nn/net"
)

func main() {
	fmt.Println("Running")
	nn := go_net.Create([]int{2, 10, 2})
	for i := 0; i < 500000; i++ {
		randVal := rand.Float64()
		go_net.Train(&nn, [][]float64{{randVal, randVal * 0.5}}, [][]float64{{randVal * randVal, randVal / 2}})
	}
	output := go_net.Run(&nn, []float64{0.4, 0.2})
	fmt.Println(output)
}
