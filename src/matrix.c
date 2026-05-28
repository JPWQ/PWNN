#include "matrix.h"

float dotProduct(float* u, float* v, int n) {
  float result = 0;
  for(int i = 0; i < n; i++) {
    result += u[i] * v[i];
  }
  return result;
}

float sigmoidf(float n) {
  return (1.0f / (1.0f + exp(-n)));
}

float reluf(float n) {
  return n <= 0 ? 0 : n;
}

float softmaxf(float n) {
  return n;
}

float dsigmoidf(float n) {
  return (n * (1.0f - n));
}

float dreluf(float n) {
  return n <= 0 ? 0 : 1;
}

float dsoftmaxf(float n) {
  return n;
}

float* matrixMult(float** weightsMatrix, float* inputVector, int rows, int cols) {
  // weights matrix should be rows x cols and input vector should be cols x 1
  float* output = malloc(sizeof(float) * rows);
  for(int i = 0; i < rows; i++) {
    float result = 0;
    for(int j = 0; j < cols; j++) {
      result += weightsMatrix[i][j] * inputVector[j];
    }
    output[i] = result;
  }
  return output;
}
