package net

import (
	layers "./layers"
)

func Create(layerOption []int) error {
	layers.Init(layerOption)
	return nil
}
