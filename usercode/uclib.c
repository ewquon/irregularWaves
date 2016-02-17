/*
 * Based on example from Steve portal:
 * "How can Brownian motion be modeled in STAR-CCM+?"
 */
#include "uclib.h"

void randomPhase(Real*,int);
void randomAmplitude(Real*,int);

void uclib()
{
    // Register user functions here

    ucfunc(randomPhase, "", "");
    ucarg(randomPhase, "", "$freq", sizeof(Real));

    ucfunc(randomAmplitude, "", "");
    ucarg(randomAmplitude, "", "$freq", sizeof(Real));
}
