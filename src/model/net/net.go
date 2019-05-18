package go_net

import (
	layers "./layers"
)

type Net struct {
	NLayers []layers.NormalLayers
}

func Create() Net {
	net := Net{}
	return net
}

func AddLayer(nn *Net, layerOption []int) {
	nn.NLayers = append(nn.NLayers, layers.NormalLayers{})
	nn.NLayers[len(nn.NLayers)-1].Lstep_main = 0.2
	// fmt.Println("init layers", nLayer.Layers)
	layers.Init(layerOption, &(nn.NLayers[len(nn.NLayers)-1]))
}

func Train(nn *Net, inputs [][]float64, outputs [][]float64) []float64 {
	if len(inputs) != len(outputs) {
		panic("Input and output lengths have to be the same for training")
	}
	for i := 0; i < len(inputs); i++ {
		Run(nn, inputs[i])
		dCostdRes := layers.BackProp(outputs[i], &(nn.NLayers[len(nn.NLayers)-1]))
		for x := len(nn.NLayers) - 2; x >= 0; x-- {
			dCostdRes = layers.BackProp_StartingCostDerriv(&(nn.NLayers[x]), dCostdRes)
		}
	}
	// fmt.Println(Layers)
	return nil
}

func Run(nn *Net, input []float64) []float64 {
	layers.RunForward(&(nn.NLayers[0]), input)
	runOut := nn.NLayers[0].CurrVals[len(nn.NLayers[0].CurrVals)-1]
	for i := 1; i < len(nn.NLayers); i++ {
		layers.RunForward(&(nn.NLayers[i]), runOut)
		runOut = nn.NLayers[i].CurrVals[len(nn.NLayers[i].CurrVals)-1]
	}
	return runOut
}

func Set_lstep(nn *Net, lstepNew float64, layer int) {
	nn.NLayers[layer].Lstep_main = lstepNew
}
