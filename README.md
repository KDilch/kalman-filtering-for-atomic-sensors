# Project description:

This repository contains the simulation of an atomic sensor that is described by a hybrid continuous-discrete probabilistic state space model. Such a sensor consists of ensemble of atoms placed in a magnetic field pumped optically to some state. The readout with the off-resonant linearly polarized beam detects the chages of one of the spin components (due to optical Faraday effect).
In this system the signal in encoded in the quadratures and this work is a prelude to a frequency extraction problem, showing that the Kalman Filters can be efficiently applied to an atomic sensor even if the dynamical model is only partially known and is highly non-linear. This repository contains the implementation of Continuous-Discrete Kalman Filter along with it's non-linear suboptimal versions (Extended Kalman Filter and Unscented Kalman Filter) as well as the architecture for the space state model that was later used for other projects.

