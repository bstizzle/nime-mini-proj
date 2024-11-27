#include <WiiChuck.h>

Accessory nunchuck1;
Accessory nunchuck2;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.setTimeout(1);

  nunchuck1.begin();
  nunchuck2.begin();
  /** If the device isn't auto-detected, set the type explicatly
  * 	NUNCHUCK,
  WIICLASSIC,
  GuitarHeroController,
  GuitarHeroWorldTourDrums,
  DrumController,
  DrawsomeTablet,
  Turntable
  */
  if (nunchuck1.type == Unknown) {
		nunchuck1.type = NUNCHUCK;
	}
  if (nunchuck2.type == Unknown) {
		nunchuck2.type = NUNCHUCK;
	}
}

void loop() {
  // put your main code here, to run repeatedly:
  nunchuck1.readData();
  nunchuck2.readData();
  
  nunchuck1.printInputs(); // Print all inputs
	for (int i = 0; i < WII_VALUES_ARRAY_SIZE; i++) {
		Serial.println(
				"Controller Val " + String(i) + " = "
						+ String((uint8_t) nunchuck1.values[i]));
	}
	nunchuck2.printInputs(); // Print all inputs
	for (int i = 0; i < WII_VALUES_ARRAY_SIZE; i++) {
		Serial.println(
				"Controller Val " + String(i) + " = "
						+ String((uint8_t) nunchuck2.values[i]));
	}

  delay(100);
}
