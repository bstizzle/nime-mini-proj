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
#include <WiiChuck.h>
#include <WiiChuckOne.h>

Accessory nunchuck1;
AccessoryOne nunchuck2;

// Neural network setup and functions START
const size_t nInstInputs=8;
const size_t nInstOutputs=3;
const size_t patternInstElements=10;
const size_t patternInstSize = patternInstElements * nInstInputs;

const unsigned int layersInst[] = {patternInstSize, 20, 20, nInstOutputs}; // 3 layers (1st)layer with 3 input neurons (2nd)layer 5 hidden neurons each and (3rd)layer with 1 output neuron
float *outputInst; 

std::vector<std::vector<float> > trainingInstInputs;
std::vector<std::vector<float> > trainingInstOutputs;

CircularBuffer<float, patternInstSize> patternInstBuffer;

enum NNMODES {TRAINING, INFERENCE};
NNMODES nnMode = NNMODES::TRAINING;

NeuralNetwork NNinst(layersInst, NumberOf(layersInst)); // Creating a Neural-Network with default learning-rates

std::vector<std::vector<float>> expectedInstOutput {
  {1,0,0}, //all violin
  {0,1,0}, //all sax
  {0,0,1}, //all pond sounds
  {0.5, 0.5, 0} //halfway between violin and sax
}; 

const size_t nLatInputs=8;
const size_t nLatOutputs=16;
const size_t patternLatElements=10;
const size_t patternLatSize = patternLatElements * nLatInputs;

const unsigned int layersLat[] = {patternLatSize, 10, 10, nLatOutputs}; // 3 layers (1st)layer with 3 input neurons (2nd)layer 5 hidden neurons each and (3rd)layer with 1 output neuron
float *outputLat; 

std::vector<std::vector<float> > trainingLatInputs;
std::vector<std::vector<float> > trainingLatOutputs;

CircularBuffer<float, patternLatSize> patternLatBuffer;

NNMODES nnLatMode = NNMODES::TRAINING;

NeuralNetwork NNlat(layersLat, NumberOf(layersLat)); // Creating a Neural-Network with default learning-rates

std::vector<std::vector<float>> expectedLatOutput {
  {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5},
  {0.5,1,0.5,1,0.5,1,0.5,1,0.5,1,0.5,1,0.5,1,0.5,1},
  {0.5,0,0.5,0,0.5,0,0.5,0,0.5,0,0.5,0,0.5,0,0.5,0},
  {1,0.5,1,0.5,1,0.5,1,0.5,1,0.5,1,0.5,1,0.5,1,0.5},
  {0,0.5,0,0.5,0,0.5,0,0.5,0,0.5,0,0.5,0,0.5,0,0.5},
  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
  {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}
}; 

void addTrainingPoint(std::vector<float> x, size_t y, String type) { //training inputs, and index to a set of outputs defined in expectedOutputs
  if(type == "i")
  {
    if (x.size() == patternInstSize) {
      trainingInstInputs.push_back(x);
      trainingInstOutputs.push_back(expectedInstOutput[y]);
      Serial.print("Instrument training point added: ");
      for(auto &v: x) {
        Serial.print(v);
        Serial.print("\t");
      }
      Serial.print(" Output: ");
      Serial.println(y);
    }else{
      Serial.print("The number of inputs should be: ");
      Serial.println(patternInstSize);
    }
  }
  else if(type == "l")
  {
    if (x.size() == patternLatSize) {
      trainingLatInputs.push_back(x);
      trainingLatOutputs.push_back(expectedLatOutput[y]);
      Serial.print("Latent training point added: ");
      for(auto &v: x) {
        Serial.print(v);
        Serial.print("\t");
      }
      Serial.print(" Output: ");
      Serial.println(y);
    }else{
      Serial.print("The number of inputs should be: ");
      Serial.println(patternLatSize);
    }
  }
}

void undoLastTraining(String type) {
  if(type == "i")
  {
    if(trainingInstInputs.size() > 0) {
      trainingInstInputs.pop_back();
      trainingInstOutputs.pop_back();
      Serial.println("Removed most recent training point");
    }else{
      Serial.println("There are no training points to remove");
    }
  }
  else if(type == "l")
  {
    if(trainingInstInputs.size() > 0) {
      trainingInstInputs.pop_back();
      trainingInstOutputs.pop_back();
      Serial.println("Removed most recent training point");
    }else{
      Serial.println("There are no training points to remove");
    }
  }
}

void train(String type) {
  size_t maxEpochs = 500;
  if(type == "i")
  {
    do{ 
      for (unsigned int j = 0; j < trainingInstInputs.size(); j++) // Epoch
      {
        NNinst.FeedForward(trainingInstInputs[j].data());      // FeedForwards the input arrays through the NN | stores the output array internally
        NNinst.BackProp(trainingInstOutputs[j].data()); // "Tells" to the NN if the output was the-expected-correct one | then, "teaches" it
      }
      
      // Prints the Error.
      Serial.print("MSE: "); 
      // Serial.println(NN.MeanSqrdError,6);
      Serial.println(NNinst.CategoricalCrossEntropy,6);
    }while(NNinst.getCategoricalCrossEntropy(trainingInstInputs.size()) > 0.0001 && maxEpochs-- > 0);
  }
  else if(type == "l")
  {
    do{ 
      for (unsigned int j = 0; j < trainingLatInputs.size(); j++) // Epoch
      {
        NNlat.FeedForward(trainingLatInputs[j].data());      // FeedForwards the input arrays through the NN | stores the output array internally
        NNlat.BackProp(trainingLatOutputs[j].data()); // "Tells" to the NN if the output was the-expected-correct one | then, "teaches" it
      }
      
      // Prints the Error.
      Serial.print("MSE: "); 
      // Serial.println(NN.MeanSqrdError,6);
      Serial.println(NNlat.CategoricalCrossEntropy,6);
    }while(NNlat.getCategoricalCrossEntropy(trainingLatInputs.size()) > 0.0001 && maxEpochs-- > 0);
  }
}

void resetTraining(String type) {
  if(type == "i")
  {
    trainingInstInputs.clear();
    trainingInstOutputs.clear();
    Serial.println("Reset instrument training data");
  }
  else if(type == "l")
  {
    trainingLatInputs.clear();
    trainingLatOutputs.clear();
    Serial.println("Reset latent training data");
  }
}

void resetModel(String type) {
  if(type == "i")
  {
    NNinst = NeuralNetwork(layersInst, NumberOf(layersInst)); // Creating a Neural-Network with default learning-rates
  }
  else if(type == "l")
  {
    NNlat = NeuralNetwork(layersLat, NumberOf(layersLat)); // Creating a Neural-Network with default learning-rates
  }
}

void printTrainingData(String type) {
  if(type == "i")
  {
    Serial.println("Instrument training data:");
    for(size_t i=0; i < trainingInstInputs.size(); i++) {
      Serial.print(i);
      Serial.print(": ");
      for(size_t j=0; j < patternInstSize; j++) {
        Serial.print(trainingInstInputs[i][j]);
        Serial.print("\t");
      }
      Serial.print(" :: ");
      for(size_t j=0; j < nInstOutputs; j++) {
        Serial.print(trainingInstOutputs[i][j]);
        Serial.print("\t");
      }
      Serial.println("");
    }
  }
  else if(type == "l")
  {
    Serial.println("Latent training data:");
    for(size_t i=0; i < trainingLatInputs.size(); i++) {
      Serial.print(i);
      Serial.print(": ");
      for(size_t j=0; j < patternLatSize; j++) {
        Serial.print(trainingLatInputs[i][j]);
        Serial.print("\t");
      }
      Serial.print(" :: ");
      for(size_t j=0; j < nLatOutputs; j++) {
        Serial.print(trainingLatOutputs[i][j]);
        Serial.print("\t");
      }
      Serial.println("");
    }
  }
}
// Neural network setup and functions END

// Hardware initializations START
// instrument sensors
int elbowLpin = 41;
int elbowRpin = 40;
int shoulderLpin = 27; //actual port TBD
int shoulderRpin = 38; //actual port TBD
int elbowL;
int elbowR;
int shoulderL;
int shoulderR;

int chipSelect = BUILTIN_SDCARD;

void setup() {
  // put your setup code here, to run once:
  pinMode(elbowLpin, INPUT);
  pinMode(elbowRpin, INPUT);
  pinMode(shoulderLpin, INPUT);
  pinMode(shoulderRpin, INPUT);

  if (!SD.begin(chipSelect)) {
    Serial.println("Could not start SD card");
  }

  Serial.begin(115200);

  nunchuck1.begin();
  nunchuck1.type = NUNCHUCK;
  nunchuck2.begin();
  nunchuck2.type = NUNCHUCKONE;
  

  Serial.setTimeout(1);
}

std::vector<float> pInst(patternInstSize);
std::vector<float> pLat(patternLatSize);
void loop() {
  // put your main code here, to run repeatedly:
  elbowL = analogRead(elbowLpin);
  elbowR = analogRead(elbowRpin);
  shoulderL = analogRead(shoulderLpin);
  shoulderR = analogRead(shoulderRpin);
  nunchuck1.readData();
  nunchuck2.readData();

  patternInstBuffer.push(elbowL/1023.0);
  patternInstBuffer.push(elbowR/1023.0);
  patternInstBuffer.push(shoulderL/1023.0);
  patternInstBuffer.push(shoulderR/1023.0);
  patternInstBuffer.push(nunchuck1.values[4]/255.0);
  patternInstBuffer.push(nunchuck1.values[5]/255.0);
  patternInstBuffer.push(nunchuck2.values[4]/255.0);
  patternInstBuffer.push(nunchuck2.values[5]/255.0);
  patternInstBuffer.copyToArray(pInst.data());
  
  patternLatBuffer.push(nunchuck1.values[0]/255.0);
  patternLatBuffer.push(nunchuck1.values[1]/255.0);
  patternLatBuffer.push(nunchuck1.values[10]/255.0);
  patternLatBuffer.push(nunchuck1.values[11]/255.0);
  patternLatBuffer.push(nunchuck2.values[0]/255.0);
  patternLatBuffer.push(nunchuck2.values[1]/255.0);
  patternLatBuffer.push(nunchuck2.values[10]/255.0);
  patternLatBuffer.push(nunchuck2.values[11]/255.0);
  
  patternLatBuffer.copyToArray(pLat.data());

  if (Serial.available()) {
    /*
    byte byteCommand = Serial.read();
    //Serial.println(byteCommand);
    char charCommand = (char)byteCommand;
    //Serial.println(charCommand);
    String command = String(charCommand);
    */
    String inputCommand = Serial.readString();
    //String command = Serial.readString();
    if (inputCommand != "") {
      String command = inputCommand[0]; //strip out \n
      String type = inputCommand[1];
      Serial.print(command);
      Serial.println(type);
      if (command == "t") { //train
        if(type == "i")
        {
          train("i");
        }
        else if(type == "l")
        {
          train("l");
        }
      }
      else if (command == "i") { //toggle inference
        if(type == "i")
        {
          if (nnMode == NNMODES::TRAINING) {
            nnMode = NNMODES::INFERENCE;
            Serial.println("Mode: Instrument Inference");
          }else {
            nnMode = NNMODES::TRAINING;
            Serial.println("Mode: Instrument Training");
          }
        }
        else if(type == "l")
        {
          if (nnLatMode == NNMODES::TRAINING) {
            nnLatMode = NNMODES::INFERENCE;
            Serial.println("Mode: Latent Inference");
          }else {
            nnLatMode = NNMODES::TRAINING;
            Serial.println("Mode: Latent Training");
          }
        }
      }
      else if (command == "s") { //status
        if(type == "i")
        {
          printTrainingData("i");
        }
        else if(type == "l")
        {
          printTrainingData("l");
        }
      }
      else if (command == "r") { //reset training
        if(type == "i")
        {
          resetTraining("i");
        }
        else if(type == "l")
        {
          resetTraining("l");
        }
      }
      else if (command == "m") { //reset model
        if(type == "i")
        {
          resetModel("i");
        }
        else if(type == "l")
        {
          resetModel("l");
        }
      }
      else if (command == "u") { //undo last data point
        if(type == "i")
        {
          undoLastTraining("i");   
        }     
        else if(type == "l")
        {
          undoLastTraining("l");   
        }
      }    
      else if (command == "c") { //save to SD
        if(type == "i")
        {
          SD.remove("inst-neural.net");
          bool res = NNinst.save("inst-neural.net");
          Serial.print("Network saved: ");
          Serial.println(res);
          NNinst.print();
        }
        else if(type == "l")
        {
          SD.remove("lat-neural.net");
          bool res = NNlat.save("lat-neural.net");
          Serial.print("Network saved: ");
          Serial.println(res);
          NNlat.print();
        }
      } 
      else if (command == "l") { //load from SD
        if(type == "i")
        {
          bool res = NNinst.load("inst-neural.net");
          Serial.print("Network loaded: ");
          Serial.println(res);
          NNinst.print();
        }
        else if(type == "l")
        {
          bool res = NNlat.load("lat-neural.net");
          Serial.print("Network loaded: ");
          Serial.println(res);
          NNlat.print();
        }
      }
      else if (isDigit(command[0])) {
        if(type == "i")
        {
          addTrainingPoint(pInst, command.toInt(), "i");
        }
        else if(type == "l")
        {
          addTrainingPoint(pLat, command.toInt(), "l");
        }
      }    
    }
  }

  if (nnMode == NNMODES::INFERENCE) {
    outputInst = NNinst.FeedForward(pInst.data());

    /*
    Serial.print("Violin level: ");
    Serial.print(output[0]);
    Serial.print(" & Sax level: ");
    Serial.println(output[1]);
    */
    
    for(size_t j=0; j < nInstOutputs; j++) {
      Serial.print(outputInst[j], 7);       // Prints the first 7 digits after the comma.
      Serial.print("\t");
    }
    Serial.println(" ");
  }
  
  if (nnLatMode == NNMODES::INFERENCE) {
    outputInst = NNinst.FeedForward(pInst.data());
    outputLat = NNlat.FeedForward(pLat.data());
    
    for(size_t j=0; j < nInstOutputs; j++) {
      Serial.print(outputInst[j], 7);       // Prints the first 7 digits after the comma.
      Serial.print("\t");
    }
    for(size_t j=0; j < nLatOutputs; j++) {
      Serial.print(outputLat[j], 7);       // Prints the first 7 digits after the comma.
      Serial.print("\t");
    }
    Serial.println(" ");
  }
  
  
  Serial.print("LE: ");
  Serial.println(elbowL);
  Serial.print("RE: ");
  Serial.println(elbowR);
  Serial.print("LS: ");
  Serial.println(shoulderL);
  Serial.print("RS: ");
  Serial.println(shoulderR);
	nunchuck1.printInputs(); // Print all inputs
  nunchuck2.printInputs();
  
  delay(10);

}
