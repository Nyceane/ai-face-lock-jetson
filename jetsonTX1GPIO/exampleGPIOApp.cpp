// exampleApp.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include "jetsonGPIO.h"
using namespace std;

int main(int argc, char *argv[]){

    cout << "Testing the GPIO Pins" << endl;


    jetsonTX1GPIONumber redLED = gpio219 ;     // Ouput

    gpioExport(redLED) ;

    gpioSetDirection(redLED,outputPin) ;

    gpioSetValue(redLED, on);
    usleep(20000000);
    gpioSetValue(redLED, off);

    cout << "GPIO example finished." << endl;
    gpioUnexport(redLED);     // unexport the LED
    return 0;
}


