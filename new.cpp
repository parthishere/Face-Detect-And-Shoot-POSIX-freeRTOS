#include <iostream>
// for delay function.
#include <chrono>
#include <thread>

// for signal handling
#include <signal.h>

#include <JetsonGPIO.h>

#define NANOSEC_TO_MICROSECOND 1000
#define SERVO1 18
#define SERVO2 12

// Pin Definitions
int output_pin = 18; // BOARD pin 12, BCM pin 18

using namespace std;

static bool end_this_program = false;

inline void delay_ns(int ns) { this_thread::sleep_for(chrono::nanoseconds(ns)); }

void signalHandler(int s) { end_this_program = true; }


void change_servo_degree(uint8_t degree){
    if(percentage > 180){
        return;
    }
    

    int turnon_us = (percentage * 1000000) / 180;
    turnon_us += 1000000;
    int turnoff_us = (2000000) - turnon_us;
    GPIO::output(output_pin, GPIO::HIGH);
    delay_ns(turnon_us);
    GPIO::output(output_pin, GPIO::LOW);
    delay_ns(turnoff_us);
}


int main()
{
    // When CTRL+C pressed, signalHandler will be called
    signal(SIGINT, signalHandler);



    // Pin Setup.
    GPIO::setmode(GPIO::BCM);
    // set pin as an output pin with optional initial state of HIGH
    GPIO::setup(output_pin, GPIO::OUT, GPIO::HIGH);

    cout << "Strating demo now! Press CTRL+C to exit" << endl;
    int curr_value = GPIO::HIGH;
    int degree = 0;
    while (!end_this_program)
    {
        change_servo_degree(degree);
        delay_ns(1000000000);
        degree++;
        if(degree == 100) {
            degree = 0;
        }
    }

    GPIO::cleanup();

    return 0;
}