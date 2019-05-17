package layers

import (
	"fmt"
	"math"
)

type NormalLayers struct {
	Layers     [][][]float64
	CurrVals   [][]float64
	CurrSums   [][]float64
	lstep_main float64
}

/* outer layer is an arr of layers. Two inners is layer. Last val of every inner is bias */
var Layers = make([][][]float64, 0)
var CurrVals = make([][]float64, 0)
var CurrSums = make([][]float64, 0)

const lstep_main = 0.1

// sigmoid as default
var activation = func(sum float64) float64 {
	return 1.0 / (1.0 + math.Exp(-sum))
}

func Init(layerOption []int) error {
	for i, _ := range layerOption {
		if i == 0 {
			Layers = append(Layers, make([][]float64, layerOption[i]))
		} else {
			Layers = append(Layers, makeNullLayer(layerOption[i], i))
		}
	}
	// fmt.Println("init layers", Layers)
	// test

	for i := 0; i < 5000; i++ {
		RunForward([]float64{float64(i), float64(i)})
		BackProp(lstep_main, []float64{0.9, 0.6})
	}
	// fmt.Println(Layers)
	RunForward([]float64{10, 30})
	fmt.Println("curr outputs", CurrVals[len(CurrVals)-1])

	return nil
}

func CalculateCost(expected []float64) float64 {
	costVal := 0.0
	lastLayer := CurrVals[len(CurrVals)-1]
	for i, neuronRes := range lastLayer {
		costVal += math.Pow((neuronRes - expected[i]), 2.0)
	}
	return costVal
}

func BackProp(lrate float64, expected []float64) error {
	if len(CurrVals[len(CurrVals)-1]) != len(expected) {
		panic("expected has to be proper length")
	}
	// cost := CalculateCost(expected)
	// for endLayerNeuron := len(expected) - 1; endLayerNeuron >= 0; endLayerNeuron-- {
	result := 0.0
	dTotaldRes := 0.0
	dResdSum := 0.0
	for layerNumb := len(Layers) - 1; layerNumb >= 1; layerNumb-- {
		newdTotaldRes := 0.0
		for i, neurons := range Layers[layerNumb] {
			if layerNumb == len(Layers)-1 {
				result = CurrVals[layerNumb][i]
				dTotaldRes = 2 * (result - expected[i])
				dResdSum = result * (1 - result)
			}
			for x, weight := range neurons {
				// update offset
				if x == len(Layers[layerNumb][i])-1 {
					dSumdB := 1.0
					dTotaldB := dSumdB * dResdSum * dTotaldRes
					Layers[layerNumb][i][x] = Layers[layerNumb][i][x] - lrate*dTotaldB
				} else {
					dSumdWeight := CurrVals[layerNumb-1][x] // Issue?
					dTotaldWeight := dSumdWeight * dResdSum * dTotaldRes
					// fmt.Println(dTotaldWeight, dTotaldRes, result, dResdSum, dSumdWeight)
					Layers[layerNumb][i][x] = weight - lrate*dTotaldWeight
					newdTotaldRes += dSumdWeight * dResdSum * weight
				}
			}
		}
		dTotaldRes = newdTotaldRes
	}
	// }
	return nil
}

func RunForward(inputs []float64) error {
	currVals := make([][]float64, len(Layers))
	currSums := make([][]float64, len(Layers))

	for i, _ := range Layers {
		currVals[i] = make([]float64, len(Layers[i]))
		currSums[i] = make([]float64, len(Layers[i]))
		if i == 0 {
			currVals[i] = inputs
		} else if i == 1 {
			for x, _ := range Layers[i] {
				sum := 0.0
				for y, _ := range Layers[i][x] {
					if y < len(Layers[i][x])-1 {
						sum += Layers[i][x][y] * inputs[y]
					} else {
						sum += Layers[i][x][y]
					}
				}
				currSums[i][x] = sum
				currVals[i][x] = activation(sum)
			}
		} else {
			for x, _ := range Layers[i] {
				sum := 0.0
				for y, _ := range Layers[i][x] {
					if y < len(Layers[i][x])-1 {
						sum += Layers[i][x][y] * currVals[i-1][y]
					} else {
						sum += Layers[i][x][y]
					}
				}
				currSums[i][x] = sum
				currVals[i][x] = activation(sum)
			}
		}
	}
	CurrVals = currVals
	CurrSums = currSums
	// fmt.Println("curr values", CurrVals)

	return nil
}

func makeNullLayer(nodeCount int, layerNumb int) [][]float64 {
	if layerNumb < 1 || layerNumb > len(Layers)+1 {
		fmt.Println(layerNumb)
		panic("wrong function use")
	}
	layer := make([][]float64, nodeCount)
	for i, _ := range layer {
		// plus one for bias!
		layer[i] = make([]float64, len(Layers[layerNumb-1])+1)
		for x, _ := range layer[i] {
			layer[i][x] = 0.01
		}
	}
	return layer
}

func getLayer(layerNumb int) [][]float64 {
	return Layers[layerNumb]
}

func setLayer(layerNumb int, layer [][]float64) error {
	Layers[layerNumb] = layer
	return nil
}
