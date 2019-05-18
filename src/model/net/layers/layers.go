package layers

import (
	"fmt"
	"math"
)

type NormalLayers struct {
	Layers     [][][]float64
	CurrVals   [][]float64
	CurrSums   [][]float64
	Lstep_main float64
}

/* outer layer is an arr of layers. Two inners is layer. Last val of every inner is bias */
// var Layers = make([][][]float64, 0)
// var CurrVals = make([][]float64, 0)
// var CurrSums = make([][]float64, 0)

const lstep_main = 0.1

// sigmoid as default
var activation = func(sum float64) float64 {
	return 1.0 / (1.0 + math.Exp(-sum))
}

func Init(layerOption []int, layersInst *NormalLayers) error {
	for i, _ := range layerOption {
		if i == 0 {
			(*layersInst).Layers = append((*layersInst).Layers, make([][]float64, layerOption[i]))
		} else {
			(*layersInst).Layers = append((*layersInst).Layers, makeNullLayer(layerOption[i], i, &((*layersInst).Layers)))
		}
	}
	// fmt.Println("init layers", Layers)
	// test

	// for i := 0; i < 5000; i++ {
	// 	RunForward([]float64{float64(i), float64(i)}, Layers)
	// 	BackProp(lstep_main, []float64{0.9, 0.6}, Layers)
	// }
	// // fmt.Println(Layers)
	// RunForward([]float64{10, 30})
	// fmt.Println("curr outputs", CurrVals[len(CurrVals)-1])

	return nil
}

func CalculateCost(expected []float64, layersInst *NormalLayers) float64 {
	costVal := 0.0
	lastLayer := layersInst.CurrVals[len(layersInst.CurrVals)-1]
	for i, neuronRes := range lastLayer {
		costVal += math.Pow((neuronRes - expected[i]), 2.0)
	}
	return costVal
}

// returns dTotaldRes for the first layer
func BackProp(expected []float64, layersInst *NormalLayers) float64 {
	lrate := layersInst.Lstep_main
	if len(layersInst.CurrVals[len(layersInst.CurrVals)-1]) != len(expected) {
		panic("expected has to be proper length")
	}
	result := 0.0
	dTotaldRes := 0.0
	dResdSum := 0.0
	for layerNumb := len(layersInst.Layers) - 1; layerNumb >= 1; layerNumb-- {
		newdTotaldRes := 0.0
		for i, neurons := range layersInst.Layers[layerNumb] {
			if layerNumb == len(layersInst.Layers)-1 {
				result = layersInst.CurrVals[layerNumb][i]
				dTotaldRes = (result - expected[i]) * 2
			}
			dResdSum = layersInst.CurrVals[layerNumb][i] * (1 - layersInst.CurrVals[layerNumb][i])
			for x, weight := range neurons {
				// update offset
				if x == len(layersInst.Layers[layerNumb][i])-1 {
					dSumdB := 1.0
					dTotaldB := dSumdB * dResdSum * dTotaldRes
					layersInst.Layers[layerNumb][i][x] = layersInst.Layers[layerNumb][i][x] - lrate*dTotaldB
				} else {
					dSumdWeight := layersInst.CurrVals[layerNumb-1][x] // Issue?
					dTotaldWeight := dSumdWeight * dResdSum * dTotaldRes
					// fmt.Println(dTotaldWeight, dTotaldRes, result, dResdSum, dSumdWeight)
					layersInst.Layers[layerNumb][i][x] = weight - lrate*dTotaldWeight
					newdTotaldRes += dTotaldRes * dResdSum * weight // weight is dSumdRes(-1)
				}
			}
		}
		dTotaldRes = newdTotaldRes
	}
	return dTotaldRes
}

func BackProp_StartingCostDerriv(layersInst *NormalLayers, dTotaldRes float64) float64 {
	lrate := layersInst.Lstep_main
	dResdSum := 0.0
	for layerNumb := len(layersInst.Layers) - 1; layerNumb >= 1; layerNumb-- {
		newdTotaldRes := 0.0
		for i, neurons := range layersInst.Layers[layerNumb] {
			dResdSum = layersInst.CurrVals[layerNumb][i] * (1 - layersInst.CurrVals[layerNumb][i])
			for x, weight := range neurons {
				// update offset
				if x == len(layersInst.Layers[layerNumb][i])-1 {
					dSumdB := 1.0
					dTotaldB := dSumdB * dResdSum * dTotaldRes
					layersInst.Layers[layerNumb][i][x] = layersInst.Layers[layerNumb][i][x] - lrate*dTotaldB
				} else {
					dSumdWeight := layersInst.CurrVals[layerNumb-1][x] // Issue?
					dTotaldWeight := dSumdWeight * dResdSum * dTotaldRes
					// fmt.Println(dTotaldWeight, dTotaldRes, result, dResdSum, dSumdWeight)
					layersInst.Layers[layerNumb][i][x] = weight - lrate*dTotaldWeight
					newdTotaldRes += dTotaldRes * dResdSum * weight // weight is dSumdRes(-1)
				}
			}
		}
		dTotaldRes = newdTotaldRes
	}
	return dTotaldRes
}

func RunForward(layersInst *NormalLayers, inputs []float64) error {
	currVals := make([][]float64, len(layersInst.Layers))
	currSums := make([][]float64, len(layersInst.Layers))

	for i, _ := range layersInst.Layers {
		currVals[i] = make([]float64, len(layersInst.Layers[i]))
		currSums[i] = make([]float64, len(layersInst.Layers[i]))
		if i == 0 {
			currVals[i] = inputs
		} else if i == 1 {
			for x, _ := range layersInst.Layers[i] {
				sum := 0.0
				for y, _ := range layersInst.Layers[i][x] {
					if y < len(layersInst.Layers[i][x])-1 {
						sum += layersInst.Layers[i][x][y] * inputs[y]
					} else {
						sum += layersInst.Layers[i][x][y]
					}
				}
				currSums[i][x] = sum
				currVals[i][x] = activation(sum)
			}
		} else {
			for x, _ := range layersInst.Layers[i] {
				sum := 0.0
				for y, _ := range layersInst.Layers[i][x] {
					if y < len(layersInst.Layers[i][x])-1 {
						sum += layersInst.Layers[i][x][y] * currVals[i-1][y]
					} else {
						sum += layersInst.Layers[i][x][y]
					}
				}
				currSums[i][x] = sum
				currVals[i][x] = activation(sum)
			}
		}
	}
	layersInst.CurrVals = currVals
	layersInst.CurrSums = currSums
	// fmt.Println("curr values", CurrVals)

	return nil
}

func makeNullLayer(nodeCount int, layerNumb int, Layers *[][][]float64) [][]float64 {
	if layerNumb < 1 || layerNumb > len((*Layers))+1 {
		fmt.Println(layerNumb)
		panic("wrong function use")
	}
	layer := make([][]float64, nodeCount)
	for i, _ := range layer {
		// plus one for bias!
		layer[i] = make([]float64, len((*Layers)[layerNumb-1])+1)
		for x, _ := range layer[i] {
			layer[i][x] = 0.01
		}
	}
	return layer
}

func GetLayer(layerNumb int, Layers *[][][]float64) [][]float64 {
	return (*Layers)[layerNumb]
}

func SetLayer(layerNumb int, layer [][]float64, Layers *[][][]float64) error {
	(*Layers)[layerNumb] = layer
	return nil
}
