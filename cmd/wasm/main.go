//go:build js && wasm

package main

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strconv"
	"strings"
	"syscall/js"
	"time"
)

var count = 0

type ActivationFunction struct {
	Activate   func(float64) float64
	Derivative func(float64) float64
}

var ActivationFunctions = map[string]ActivationFunction{
	"sigmoid": {
		Activate: func(x float64) float64 { return 1 / (1 + math.Exp(-x)) },
		Derivative: func(x float64) float64 {
			sig := 1 / (1 + math.Exp(-x))
			return sig * (1 - sig)
		},
	},
	"relu": {
		Activate: func(x float64) float64 { return math.Max(0, x) },
		Derivative: func(x float64) float64 {
			if x > 0 {
				return 1
			} else {
				return 0
			}
		},
	},
	"tanh": {
		Activate:   func(x float64) float64 { return math.Tanh(x + 1e-15) },
		Derivative: func(x float64) float64 { return 1 - math.Pow(math.Tanh(x+1e-15), 2) },
	},
	"linear": {
		Activate:   func(x float64) float64 { return x },
		Derivative: func(x float64) float64 { return 1 },
	},
}

type Neuron struct {
	Weights []float64
	Bias    float64
}

type Layer struct {
	ActivateFunction ActivationFunction
	Neurons          []Neuron
}

type MLP struct {
	ClassMap     map[string]int
	InvClassMap  map[int]string
	Layers       []Layer
	LearningRate float64
}

func NewMLP(inputSize int, hiddenSizes []int, outputSize int, learningRate float64, activations []string) *MLP {
	layers := make([]Layer, len(hiddenSizes)+1)
	prevSize := inputSize

	for i, size := range hiddenSizes {
		layers[i] = Layer{
			Neurons:          make([]Neuron, size),
			ActivateFunction: ActivationFunctions[activations[i]],
		}
		for j := range layers[i].Neurons {
			layers[i].Neurons[j] = Neuron{
				Weights: make([]float64, prevSize),
				Bias:    rand.Float64()*2 - 1,
			}
			for k := range layers[i].Neurons[j].Weights {
				layers[i].Neurons[j].Weights[k] = rand.Float64()*2 - 1
			}
		}
		prevSize = size
	}

	layers[len(layers)-1] = Layer{
		Neurons:          make([]Neuron, outputSize),
		ActivateFunction: ActivationFunctions[activations[len(activations)-1]],
	}
	for j := range layers[len(layers)-1].Neurons {
		layers[len(layers)-1].Neurons[j] = Neuron{
			Weights: make([]float64, prevSize),
			Bias:    rand.Float64()*2 - 1,
		}
		for k := range layers[len(layers)-1].Neurons[j].Weights {
			layers[len(layers)-1].Neurons[j].Weights[k] = rand.Float64()*2 - 1
		}
	}

	return &MLP{
		Layers:       layers,
		LearningRate: learningRate,
		ClassMap:     make(map[string]int),
		InvClassMap:  make(map[int]string),
	}
}

func (mlp *MLP) Forward(input []float64) []float64 {
	current := input
	for _, layer := range mlp.Layers {
		next := make([]float64, len(layer.Neurons))
		for i, neuron := range layer.Neurons {
			sum := neuron.Bias
			for j, weight := range neuron.Weights {
				sum += weight * current[j]
			}
			next[i] = layer.ActivateFunction.Activate(sum)
		}
		current = next
	}
	return current
}

// NOTE: IGNORE THESE FUNCTIONS, ONLY WRITTEN FOR FUN
func (mlp *MLP) ForwardRecursive(input []float64) []float64 {
	return mlp.forwardRecursive(input, 0)
}

func (mlp *MLP) forwardRecursive(input []float64, layerIndex int) []float64 {
	if layerIndex >= len(mlp.Layers) {
		return input
	}

	layer := mlp.Layers[layerIndex]
	next := mlp.computeLayerOutput(layer, input, 0)
	return mlp.forwardRecursive(next, layerIndex+1)
}

func (mlp *MLP) computeLayerOutput(layer Layer, input []float64, neuronIndex int) []float64 {
	if neuronIndex >= len(layer.Neurons) {
		return make([]float64, len(layer.Neurons)) // Base case: All neurons have been processed
	}

	neuron := layer.Neurons[neuronIndex]
	sum := mlp.computeNeuronOutput(neuron, input, 0)
	next := mlp.computeLayerOutput(layer, input, neuronIndex+1) // Recurse for next neuron

	next[neuronIndex] = layer.ActivateFunction.Activate(sum) // Store current neuron's output
	return next
}

func (mlp *MLP) computeNeuronOutput(neuron Neuron, input []float64, weightIndex int) float64 {
	if weightIndex >= len(neuron.Weights) {
		return neuron.Bias // Base case: all weights have been processed
	}

	// Recursive summation of weights * inputs
	sum := neuron.Weights[weightIndex] * input[weightIndex]
	return sum + mlp.computeNeuronOutput(neuron, input, weightIndex+1)
}

// NOTE: END IGNORE ================================================

func (mlp *MLP) Train(inputs [][]float64, targets [][]float64, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		correctPredictions := 0

		for i, input := range inputs {
			// Forward pass
			activations := make([][]float64, len(mlp.Layers)+1)
			activations[0] = input
			for j, layer := range mlp.Layers {
				activations[j+1] = make([]float64, len(layer.Neurons))
				for k, neuron := range layer.Neurons {
					sum := neuron.Bias
					for l, weight := range neuron.Weights {
						sum += weight * activations[j][l]
					}
					activations[j+1][k] = layer.ActivateFunction.Activate(sum)
				}
			}

			// Compute loss
			output := activations[len(activations)-1]
			loss := 0.0
			for j := range output {
				diff := targets[i][j] - output[j]
				loss += diff * diff
			}
			totalLoss += loss

			// Check if prediction is correct
			predictedClass := argmax(output)
			targetClass := argmax(targets[i])
			if predictedClass == targetClass {
				correctPredictions++
			}

			// Backward pass
			deltas := make([][]float64, len(mlp.Layers))
			for j := len(mlp.Layers) - 1; j >= 0; j-- {
				deltas[j] = make([]float64, len(mlp.Layers[j].Neurons))
				if j == len(mlp.Layers)-1 {
					for k := range mlp.Layers[j].Neurons {
						deltas[j][k] = (activations[j+1][k] - targets[i][k]) * mlp.Layers[j].ActivateFunction.Derivative(activations[j+1][k])
					}
				} else {
					for k := range mlp.Layers[j].Neurons {
						sum := 0.0
						for l, neuron := range mlp.Layers[j+1].Neurons {
							sum += neuron.Weights[k] * deltas[j+1][l]
						}
						deltas[j][k] = sum * mlp.Layers[j].ActivateFunction.Derivative(activations[j+1][k])
					}
				}
			}

			// Update weights and biases
			for j, layer := range mlp.Layers {
				for k, neuron := range layer.Neurons {
					for l := range neuron.Weights {
						neuron.Weights[l] -= mlp.LearningRate * deltas[j][k] * activations[j][l]
					}
					neuron.Bias -= mlp.LearningRate * deltas[j][k]
				}
			}
		}

		accuracy := float64(correctPredictions) / float64(len(inputs))
		avgLoss := totalLoss / float64(len(inputs))
		fmt.Printf("Epoch %d: Loss = %.4f, Accuracy = %.4f\n", epoch+1, avgLoss, accuracy)
	}
}

func (mlp *MLP) Predict(input []float64) string {
	output := mlp.Forward(input)
	maxIndex := argmax(output)
	return mlp.InvClassMap[maxIndex]
}

func argmax(slice []float64) int {
	maxIndex := 0
	maxValue := slice[0]
	for i, value := range slice {
		if value > maxValue {
			maxIndex = i
			maxValue = value
		}
	}
	return maxIndex
}

func loadCSV(csvData string, skipHeader bool) ([][]float64, []string, error) {
	records := strings.Split(csvData, "\n")

	var err error

	data := records
	// Skip header
	if skipHeader {
		data = records[1:]
	}
	features := make([][]float64, len(data))
	labels := make([]string, len(data))

	for i, recordStr := range data {
		record := strings.Split(recordStr, ",")
		features[i] = make([]float64, len(record)-1)
		for j, value := range record[:len(record)-1] {
			features[i][j], err = strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, nil, err
			}
		}
		labels[i] = record[len(record)-1]
	}

	return features, labels, nil
}

func oneHotEncode(labels []string) ([][]float64, map[string]int, map[int]string) {
	classMap := make(map[string]int)
	invClassMap := make(map[int]string)
	for _, label := range labels {
		if _, exists := classMap[label]; !exists {
			index := len(classMap)
			classMap[label] = index
			invClassMap[index] = label
		}
	}

	encoded := make([][]float64, len(labels))
	for i, label := range labels {
		encoded[i] = make([]float64, len(classMap))
		encoded[i][classMap[label]] = 1
	}

	return encoded, classMap, invClassMap
}

func mlp(data string, hiddenSizes []int, learningRate float64, activations []string, epochs int) {
	// Load and preprocess data
	features, labels, err := loadCSV(data, true)
	// features, labels, err := loadCSV("heart.csv", true)
	// features, labels, err := loadCSV("iris.csv", true)
	// features, labels, err := loadCSV("assets/test.txt", true)
	if err != nil {
		fmt.Println("Error loading CSV:", err)
		return
	}

	encodedLabels, classMap, invClassMap := oneHotEncode(labels)

	inputSize := len(features[0])
	outputSize := len(classMap)

	mlp := NewMLP(inputSize, hiddenSizes, outputSize, learningRate, activations)
	mlp.ClassMap = classMap
	mlp.InvClassMap = invClassMap

	// Train the model
	mlp.Train(features, encodedLabels, epochs)

	// Make predictions
	for i := 0; i < 10; i++ {
		// get random sample
		s := rand.IntN(len(features))

		// predict
		prediction := mlp.Predict(features[s])

		correct := "Incorrect"

		if prediction == labels[s] {
			correct = "Correct"
		}

		fmt.Printf("[%s] Sample %2d - Predicted: %s, Actual: %s\n", correct, i+1, prediction, labels[s])
	}
}

func getJSValue(id string) js.Value {
	return js.Global().Get("document").Call("getElementById", id)
}

func getValue(id string) string {
	return getJSValue(id).Get("value").String()
}

func bindButton(id string, f func()) {
	getJSValue(id).Set("onclick", js.FuncOf(
		func(this js.Value, args []js.Value) any {
			f()
			return nil
		},
	))
}

var (
	dataChannel = make(chan bool)
	data        string
)

func fileInput() {
	fileInput := getJSValue("fileInput")

	fileInput.Set("oninput", js.FuncOf(func(v js.Value, x []js.Value) any {
		fileInput.Get("files").
			Call("item", 0).
			Call("arrayBuffer").
			Call(
				"then",
				js.FuncOf(
					func(v js.Value, x []js.Value) any {
						jsData := js.Global().Get("Uint8ClampedArray").New(x[0])
						dst := make([]byte, jsData.Get("length").Int())
						js.CopyBytesToGo(dst, jsData)

						out := string(dst)
						data = out

						dataChannel <- true

						return nil
					},
				),
			)

		return nil
	}))
}

func main() {
	fileInput()
	// mlp(strings.TrimSpace(readDataSet("test.txt")))

	for {
		<-dataChannel
		fileOutput := getJSValue("fileOutput")
		fileOutput.Set("innerText", data)

		// sleep for 100ms
		time.Sleep(time.Millisecond * 10)
		// Create MLP
		hiddenSizes := []int{12, 12}
		learningRate := 0.001
		activations := []string{"relu", "relu", "sigmoid"}
		epochs := 500
		mlp(strings.TrimSpace(data), hiddenSizes, learningRate, activations, epochs)
	}
}

func readDataSet(fileName string) string {
	var data string

	var fetchThen js.Value

	js.Global().Call("fetch", fileName).Call("then", js.FuncOf(
		func(v js.Value, x []js.Value) any {
			fetchThen = x[0]
			dataChannel <- true
			return nil
		},
	))

	// wait for fetch to complete
	<-dataChannel

	fetchThen.Call("text").Call("then", js.FuncOf(
		func(v js.Value, x []js.Value) any {
			data = x[0].String()
			dataChannel <- true
			return nil
		},
	))

	// wait for data to be read
	<-dataChannel

	return data
}
