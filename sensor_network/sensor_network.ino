#define NumberOf(arg) ((unsigned int) (sizeof (arg) / sizeof (arg [0]))) // calculates the number of layers (in this case 3)
#define _2_OPTIMIZE 0B00000000 // Enable 0B01.. for NO_BIAS or 0B001.. for MULTIPLE_BIASES_PER_LAYER
#define _1_OPTIMIZE 0B00010000 // https://github.com/GiorgosXou/NeuralNetworks#define-macro-properties
//#define Tanh                   // Comment this line to use Sigmoid (the default) activation function
#define CATEGORICAL_CROSS_ENTROPY
#include <NeuralNetwork.h>
#include <vector>
#include <CircularBuffer.hpp>
#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>

// Neural network setup and functions START
const size_t nInputs=4;
const size_t nOutputs=2;
const size_t patternElements=10;
const size_t patternSize = patternElements * nInputs;

const unsigned int layers[] = {patternSize, 20, 20, nOutputs}; // 3 layers (1st)layer with 3 input neurons (2nd)layer 5 hidden neurons each and (3rd)layer with 1 output neuron
float *output; 

std::vector<std::vector<float> > trainingInputs;
std::vector<std::vector<float> > trainingOutputs;

CircularBuffer<float, patternSize> patternBuffer;

enum NNMODES {TRAINING, INFERENCE};
NNMODES nnMode = NNMODES::TRAINING;

NeuralNetwork NN(layers, NumberOf(layers)); // Creating a Neural-Network with default learning-rates

std::vector<std::vector<float>> expectedOutput {
  {1,0}, //all violin
  {0,1}, //all sax
  {0.5, 0.5} //halfway between violin and sax
}; 

void addTrainingPoint(std::vector<float> x, size_t y) { //training inputs, and index to a set of outputs defined in expectedOutputs
  if (x.size() == patternSize) {
    trainingInputs.push_back(x);
    trainingOutputs.push_back(expectedOutput[y]);
    Serial.print("Training point added: ");
    for(auto &v: x) {
      Serial.print(v);
      Serial.print("\t");
    }
    Serial.print(" Output: ");
    Serial.println(y);
  }else{
    Serial.print("The number of inputs should be: ");
    Serial.println(patternSize);
  }
}

void undoLastTraining() {
  if (trainingInputs.size() > 0) {
    trainingInputs.pop_back();
    trainingOutputs.pop_back();
    Serial.println("Removed most recent training point");
  }else{
    Serial.println("There are no training points to remove");
  }
}

void train() {
  size_t maxEpochs = 500;
  do{ 
    for (unsigned int j = 0; j < trainingInputs.size(); j++) // Epoch
    {
      NN.FeedForward(trainingInputs[j].data());      // FeedForwards the input arrays through the NN | stores the output array internally
      NN.BackProp(trainingOutputs[j].data()); // "Tells" to the NN if the output was the-expected-correct one | then, "teaches" it
    }
    
    // Prints the Error.
    Serial.print("MSE: "); 
    // Serial.println(NN.MeanSqrdError,6);
    Serial.println(NN.CategoricalCrossEntropy,6);
  }while(NN.getCategoricalCrossEntropy(trainingInputs.size()) > 0.0001 && maxEpochs-- > 0);
}

void resetTraining() {
  trainingInputs.clear();
  trainingOutputs.clear();
  Serial.println("Reset training data");
}

void resetModel() {
  NN = NeuralNetwork(layers, NumberOf(layers)); // Creating a Neural-Network with default learning-rates
}

void printTrainingData() {
  Serial.println("Training data:");
  for(size_t i=0; i < trainingInputs.size(); i++) {
    Serial.print(i);
    Serial.print(": ");
    for(size_t j=0; j < patternSize; j++) {
      Serial.print(trainingInputs[i][j]);
      Serial.print("\t");
    }
    Serial.print(" :: ");
    for(size_t j=0; j < nOutputs; j++) {
      Serial.print(trainingOutputs[i][j]);
      Serial.print("\t");
    }
    Serial.println("");
  }
}
// Neural network setup and functions END

// Hardware initializations START
// instrument sensors
int elbowLpin = 41;
int elbowRpin = 40;
int shoulderLpin = 39; //actual port TBD
int shoulderRpin = 38; //actual port TBD
int elbowL;
int elbowR;
int shoulderL;
int shoulderR;

// latent space sensors
int latent1pin = 24; //actual port TBD
int latent2pin = 25; //actual port TBD
int latent1;
int latent2;

void setup() {
  // put your setup code here, to run once:
  pinMode(elbowLpin, INPUT);
  pinMode(elbowRpin, INPUT);
  pinMode(shoulderLpin, INPUT);
  pinMode(shoulderRpin, INPUT);
  pinMode(latent1pin, INPUT);
  pinMode(latent2pin, INPUT);

  Serial.begin(115200);
  Serial.setTimeout(1);
}

std::vector<float> p(patternSize);
void loop() {
  // put your main code here, to run repeatedly:
  elbowL = analogRead(elbowLpin);
  elbowR = analogRead(elbowRpin);
  shoulderL = analogRead(shoulderLpin);
  shoulderR = analogRead(shoulderRpin);
  latent1 = analogRead(latent1pin);
  latent2 = analogRead(latent2pin);

  patternBuffer.push(elbowL/1023.0);
  patternBuffer.push(elbowR/1023.0);
  patternBuffer.push(shoulderL/1023.0);
  patternBuffer.push(shoulderR/1023.0);
  patternBuffer.copyToArray(p.data());

  if (Serial.available()) {
    byte byteCommand = Serial.read();
    //Serial.println(byteCommand);
    char charCommand = (char)byteCommand;
    //Serial.println(charCommand);
    String command = String(charCommand);
    
    //String command = Serial.readString();
    if (command != "") {
      command = command[0]; //strip out \n
      Serial.println(command);
      if (command == "t") { //train
        train();
      } 
      else if (command == "i") { //toggle inference
        if (nnMode == NNMODES::TRAINING) {
          nnMode = NNMODES::INFERENCE;
          Serial.println("Mode: Inference");
        }else {
          nnMode = NNMODES::TRAINING;
          Serial.println("Mode: Training");
        }
      }    
      else if (command == "s") { //status
        printTrainingData();
      }    
      else if (command == "r") { //reset training
        resetTraining();
      }    
      else if (command == "m") { //reset model
        resetModel();
      }    
      else if (command == "u") { //undo last data point
        undoLastTraining();        
      }    
      else if (command == "c") { //save to SD
        SD.remove("neural.net");
        bool res = NN.save("neural.net");
        Serial.print("Network saved: ");
        Serial.println(res);
        NN.print();
      }    
      else if (command == "l") { //load from SD
        bool res = NN.load("neural.net");
        Serial.print("Network loaded: ");
        Serial.println(res);
        NN.print();
      }    
      else if (isDigit(command[0])) {
        addTrainingPoint(p, command.toInt());
      }    
    }
  }

  if (nnMode == NNMODES::INFERENCE) {
    output = NN.FeedForward(p.data());

    /*
    Serial.print("Violin level: ");
    Serial.print(output[0]);
    Serial.print(" & Sax level: ");
    Serial.println(output[1]);
    */
    
    for(size_t j=0; j < nOutputs; j++) {
      Serial.print(output[j], 7);       // Prints the first 7 digits after the comma.
      Serial.print("\t");
    }
    Serial.print(latent1);
    Serial.print("\t");
    Serial.print(latent2);
    Serial.print("\t");
    Serial.println("");
  }

  Serial.print("LE: ");
  Serial.println(elbowL);
  Serial.print("RE: ");
  Serial.println(elbowR);
  Serial.print("LS: ");
  Serial.println(shoulderL);
  Serial.print("RS: ");
  Serial.println(shoulderR);
  delay(100);
}
