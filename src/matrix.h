#include <math.h>
#include <stdlib.h>

float dotProduct(float* u, float* v, int n);
float sigmoidf(float n);
float reluf(float n);
float softmaxf(float n);
float dsigmoidf(float n);
float dreluf(float n);
float dsoftmaxf(float n);
float* matrixMult(float** weightsMatrix, float* outputVector, int m, int n);
