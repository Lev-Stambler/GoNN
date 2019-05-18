package main

import (
	"fmt"
	"math/rand"

	go_net "./model/net"
)

func main() {
	fmt.Println("Running")
	nn := go_net.Create()
	go_net.AddLayer(&nn, []int{1, 5})
	go_net.AddLayer(&nn, []int{5, 1})
	go_net.Set_lstep(&nn, 0.2, 0)
	for i := 0; i < 500000; i++ {
		randVal := rand.Float64()
		go_net.Train(&nn, [][]float64{{randVal}}, [][]float64{{randVal * randVal}})
	}
	output := go_net.Run(&nn, []float64{0.5})
	fmt.Println(output)

	output = go_net.Run(&nn, []float64{0.2})
	fmt.Println(output)
}
