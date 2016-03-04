# FluidSolver2D
2D fluid simulator in c++ using OpenCL based on the paper "Real-Time Fluid Dynamics for Games" by Jos Stam

![Screenshot](image/smoke.png)

# Dependencies

This project requires the SFML 2.0 and OpenCL 1.2.
The main configuration variables are located in config.h where you can change the screen resolution, the OpenCL device you want to use and the fluid properties.

# Usage

* ESC - exit the program
* Left mouse button - add density under the cursor within a certain radius
* Right mouse drag - add velocity to the velocity field in the direction of the mouse within a certain radius
* Mouse wheel - change the radius 
* Space - reset the simulation 