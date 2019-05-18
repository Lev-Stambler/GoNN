package go_net

import (
	layers "./layers"
)

type Net struct {
	NLayer layers.NormalLayers
}

func Create(layerOption []int) Net {
	net := Net{}
	net.NLayer = layers.NormalLayers{}
	net.NLayer.Lstep_main = 0.2
	// fmt.Println("init layers", nLayer.Layers)
	layers.Init(layerOption, &(net.NLayer))
	return net
}

func Train(nn *Net, inputs [][]float64, outputs [][]float64) error {
	if len(inputs) != len(outputs) {
		panic("Input and output lengths have to be the same for training")
	}
	for i := 0; i < len(inputs); i++ {
		layers.RunForward(inputs[i], &(nn.NLayer))
		layers.BackProp(outputs[i], &(nn.NLayer))
	}
	// fmt.Println(Layers)
	return nil
}

func Run(nn *Net, input []float64) []float64 {
	layers.RunForward(input, &(nn.NLayer))
	return nn.NLayer.CurrVals[len(nn.NLayer.CurrVals)-1]
}

func Set_lstep(nn *Net, lstepNew float64) {
	nn.NLayer.Lstep_main = lstepNew
}
