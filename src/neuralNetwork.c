#include "neuralNetwork.h"

Layer createLayer(int prevSize, int curSize, char* activation) {
  srand(time(NULL));
  // float weightsArr[curSize][prevSize];

  // The weights 2d array takes n float pointers where n is the number of neurons in the next/current layer.
  float** weightsArr = (float**)malloc(sizeof(float*) * curSize);

  // Each neruon has it's own bias so it must be a 1d array
  // taking the size of the number of neurons in the next/current layer.
  // I chose this because after forward and backward propagation each neruon
  // should be better optimized.
  // Bias array is a column vector
  float* biasArr = malloc(sizeof(float) * curSize);

  // Intialize the output array of the current layer.
  // This will be overwritten by the Compute Layer function.
  // Output array is a column vector
  float* outputArr = malloc(sizeof(float) * curSize);
  
  // Initialzie array for deltas used in backpropagation
  float* deltaArr = malloc(sizeof(float) * curSize);

  // Initialize the activation function
  float (*activationFunc)(float) = reluf;
  if (stringCmp(activation, "relu")) {
    activationFunc = reluf;
  }
  else if (stringCmp(activation, "sigmoid")) {
    activationFunc = sigmoidf;
  }
  else if (stringCmp(activation, "softmax")) {
    activationFunc = softmaxf;
  }
  else {
    printf("Error: Invalid Activation Function. Defaulting to Relu\n");
  }

  float (*derivativeActFunc)(float) = dreluf;
  if (activationFunc == reluf) {
    derivativeActFunc = dreluf;
  }
  else if (activationFunc == sigmoidf) {
    derivativeActFunc = dsigmoidf;
  }
  else if (activationFunc == softmaxf) {
    derivativeActFunc = dsoftmaxf;
  }
  else {
    printf("Error: Default Derivatvie Activation Function Relu\n");
  }

  // For each neuron in the next/current layer initialize
  // each element of output array is a random number greater than 0 if the curLayer is not an input layer
  // each element of the weights array is a float pointer of size of the previous layer
  // each element of the biass array is a random number greater than 0

  for(int i = 0; i < curSize; i++) {
    // If the current layer is the input layer assign values to it's output array.
    // This is neccessary because when computeLayer() is called in createNeuralNetwork()
    // it skips the first array because there is no previous layer for the input layer.
    weightsArr[i] = (float*)malloc(sizeof(float) * prevSize);

    biasArr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    // For each neuron in the previous layer initialize
    // the ith pointer to a random number greater than 0
    for(int j = 0; j < prevSize; j++) {
      weightsArr[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
  }

  Layer newLayer = {weightsArr, biasArr, outputArr, deltaArr, curSize, prevSize, activationFunc, derivativeActFunc};
  printf("Act: %p, dAct: %p, sig: %p, dsig: %p\n", newLayer.activationFunc, newLayer.derivativeActFunc, sigmoidf, dsigmoidf);
  return newLayer;
}

// Using the weights and bias from the previous layer compute the output of the current layer
void computeLayer(Layer* prevLayer, Layer* curLayer, float* inputVector) {
  if(prevLayer->inputLayer) {
    prevLayer->output = inputVector;
  }
  // Using the formula Z = W*A + B

  // Taking the transpose is unnecessary because
  // the weights matrix uses the row vector convention

  // Then using the sigmoid activation function apply it to each output in the output vector
  float* result = matrixMult(curLayer->weights, prevLayer->output, curLayer->curSize, prevLayer->curSize);
  for(int i = 0; i < curLayer->curSize; i++) {
    result[i] += curLayer->bias[i];
    result[i] = curLayer->activationFunc(result[i]);
  }
  curLayer->output = result;
}

void computeOutputDeltas(Layer* layer, float* expectedOutput) {
  for(int i = 0; i < layer->curSize; i++) {
    // error is partial derivative of Cost function (MSE) with respect to activated output
    float error = expectedOutput[i] - layer->output[i];
    // outputDeltas is error times derivative of activation function
    layer->deltas[i] = error * layer->derivativeActFunc(layer->output[i]);
  }
}

void computeHiddenDeltas(Layer* prevLayer, Layer* curLayer) {
  for(int i = 0; i < prevLayer->curSize; i++) {
    float error = 0;
    for(int j = 0; j < curLayer->curSize; j++) {
      // error is the sum of each neuron weight times the delta
      error += (curLayer->weights[j][i] * curLayer->deltas[j]);
    }
    // prev layer deltas is the derivative of the activation function times error
    prevLayer->deltas[i] = prevLayer->activationFunc(prevLayer->output[i]) * error;
  }
}

void updateWeightsAndBias(Layer* prevLayer, Layer* curLayer, float learningRate) {
  for(int i = 0; i < curLayer->curSize; i++) {
    // update the bias with the deltas times the learning rate
    curLayer->bias[i] += learningRate * curLayer->deltas[i];

    // for each neuron in the previous layer update the weights of the current layer
    // each neurons weight is the learning rate times the delta times the prev layer ouput
    for(int j = 0; j < prevLayer->curSize; j++) {
      curLayer->weights[i][j] += learningRate * curLayer->deltas[i] * prevLayer->output[j];
    }
  } 
}

void trainNeuralNetwork(NeuralNetwork neuralNetwork, float learningRate, int epochs, float** expectedOutputMatrix, float** inputMatrix) {
  // run this sequnece for each epoch
  for(int epoch = 0; epoch < epochs; epoch++) {
    // for each sample provided
    for(int sample = 0; sample < neuralNetwork.layers[0].prevSize; sample++) {
      float* inputVector = inputMatrix[sample];
      
      // compute each layer of the neural network
      for(int i = 1; i < neuralNetwork.numLayers; i++) {
        computeLayer(&neuralNetwork.layers[i - 1], &neuralNetwork.layers[i], inputVector);
      }
      
      // then compute the output deltas
      computeOutputDeltas(&neuralNetwork.layers[neuralNetwork.numLayers - 1], expectedOutputMatrix[sample]);
      
      // then compute the deltas of the hidden layers
      for(int i = neuralNetwork.numLayers - 1; i > 1 ; i--) {
        computeHiddenDeltas(&neuralNetwork.layers[i - 1], &neuralNetwork.layers[i]);
      }

      // update the weights and biases of each neuron
      for(int i = 1; i < neuralNetwork.numLayers; i++) {
        updateWeightsAndBias(&neuralNetwork.layers[i - 1], &neuralNetwork.layers[i], learningRate);
      }
    }
  }
}

void testNeuralNetwork(NeuralNetwork neuralNetwork, float* inputVector, float expectedOutput) {
  for(int i = 1; i < neuralNetwork.numLayers; i++) {
    computeLayer(&neuralNetwork.layers[i - 1], &neuralNetwork.layers[i], inputVector);
  }
  printf("Input: ");
  for(int i = 0; i < 2; i++) {
    printf("%.2f, ", inputVector[i]);
  }
  printf("Predicted Output: %f", neuralNetwork.layers[neuralNetwork.numLayers - 1].output[0]);
  printf("\n");
  printf("Error: %f\n", powf((neuralNetwork.layers[neuralNetwork.numLayers - 1].output[0] - expectedOutput), 2));

}

void freeNN(NeuralNetwork neuralNetwork) {
  for(int i = 0; i < neuralNetwork.numLayers; i++) {
    for(int j = 0; j < neuralNetwork.layers[i].curSize; j++) {
      free(neuralNetwork.layers[i].weights[j]);
    }
    free(neuralNetwork.layers[i].weights);
    free(neuralNetwork.layers[i].bias);
    if (!neuralNetwork.layers[i].inputLayer) {
      free(neuralNetwork.layers[i].output);
    }
    free(neuralNetwork.layers[i].deltas);
  }
  free(neuralNetwork.layers);
}

NeuralNetwork createNeuralNetwork(int numLayers, ...) {
  NeuralNetwork neuralNetwork = {malloc(sizeof(Layer) * numLayers), numLayers};

  va_list args;
  va_start(args, numLayers);

  for(int i = 0; i < numLayers; i++) {
    neuralNetwork.layers[i] = va_arg(args, Layer);
    if(i == 0) {
      neuralNetwork.layers[i].inputLayer = true;
    }
    else if (i == numLayers - 1) {
      neuralNetwork.layers[i].outputLayer = true;
    }
    else {
      neuralNetwork.layers[i].hiddenLayer = true;
    }
  }

  va_end(args);
  return neuralNetwork;
}
