#include <stdio.h>
#include <stdbool.h>
#include <stdarg.h>
#include "matrix.h"

// Each layer keeps track of how many neurons it has and an array of Neurons
typedef struct layer_t {
  float** weights;
  float* bias;
  float* output;
  float* deltas;
  int curSize;
  int prevSize;
  bool inputLayer;
  bool hiddenLayer;
  bool outputLayer;
} Layer;

typedef struct neuralNetwork_t {
  Layer* layers;
  int numLayers;
} NeuralNetwork;

Layer createLayer(int curSize, int nextSize);
void computeLayer(Layer* prevLayer, Layer* curLayer, float* inputVector);
void comuteOutputDeltas(Layer* layer, float* expectedOutput);
void computeHiddenDeltas(Layer* prevLayer, Layer* curLayer);
void updateWeightsAndBias(Layer* prevLayer, Layer* curLayer, float learningRate);
void trainNeuralNetwork(NeuralNetwork neuralNetwork, float learningRate, int epochs, float** expectedOutput, float** inputMatrix);
void testNeuralNetwork(NeuralNetwork neuralNetwork, float* inputVector);
void freeNN(NeuralNetwork neuralNetwork);
NeuralNetwork createNeuralNetwork(int numLayers, ...);
