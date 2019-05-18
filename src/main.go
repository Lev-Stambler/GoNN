package main

import (
	"fmt"
	"math/rand"

	go_net "./nn/net"
)

func main() {
	fmt.Println("Running")
	nn := go_net.Create([]int{1, 17, 15, 1})
	go_net.Set_lstep(&nn, 0.05)
	for i := 0; i < 500000; i++ {
		randVal := rand.Float64()
		go_net.Train(&nn, [][]float64{{randVal}}, [][]float64{{randVal * randVal / 2}})
	}
	output := go_net.Run(&nn, []float64{0.5})
	fmt.Println(output)
	// fmt.Println(nn.NLayer.CurrVals)
	// fmt.Println(nn.NLayer.Layers)

	output = go_net.Run(&nn, []float64{0.7})
	fmt.Println(output)

	output = go_net.Run(&nn, []float64{0.9})
	fmt.Println(output)
	// fmt.Println(nn.NLayer.CurrVals)
	// fmt.Println(nn.NLayer.Layers)

}
