#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H

#include <iostream>
#include <fstream>
#include <CL/cl.hpp>
#include <SFML/Graphics.hpp>

class FluidSolver
{
public:
	/** Constructor */
	FluidSolver();
	/** Destructor */
	virtual ~FluidSolver();
	/** Initialize the gpu and the internal buffer required to the simulation */
	void initialization();
	/** Update the simulation */
	void update(float dt);
	/** Add density "intensity" of fluid in the circle of radius "radius" centered at (x,y) */
	void add_pressure(int x, int y, int radius, float intensity);
	/** Add the velocity (dx,dy) vector to the velocity field in the circle of radius "radius" centered at (x,y) */
	void add_velocity(int x, int y, float dx, float dy, float force, int radius);
	/** Used to synchronize the gpu image buffer with any RGBA uint8_t array */
	void set_data_image(cl_uint8* img);
	/** Update the array "ptr" passed in the function "set_data_image(ptr)" */
	void update_image();
	/** Reset the simulation (the density and velocity fields will be set to 0 everywhere) */
	void reset();
protected:
	void cl_init();
	void program_init();
	void add_source(cl::Image2D & in_out, int x, int y, int radius, float intensity);
	void advect(cl::Image2D & dest, const cl::Image2D & src, cl::Image2D & img_u, cl::Image2D & img_v, float dt, int bound);
	void project(cl::Image2D & img_u, cl::Image2D & img_v);
	void diffuse(cl::Image2D & input_output, const cl::Image2D & src, float diff, float diff_div, int bound);
	// opencl
	std::vector<cl::Platform> all_platforms;
	cl::Platform default_platform;
	cl::Device default_device;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
	// utility variables
	cl::size_t<3> origin;
	cl::size_t<3> region;
	cl::size_t<3> regionf;
	cl::NDRange origin_work;
	cl::NDRange region_work;
	cl::NDRange origin_work_center;
	cl::NDRange region_work_center;
	// opencl kernels
	cl::Kernel kernel_diffuse;
	cl::Kernel kernel_advect;
	cl::Kernel kernel_project1;
	cl::Kernel kernel_project2;
	cl::Kernel kernel_reset;
	cl::Kernel kernel_addsource;
	cl::Kernel kernel_draw_img;
	// gpu memory structures
	cl_uint8* data_image;// pointer on the sfml image memory
	cl::Image2D density_in;
	cl::Image2D density_out;
	cl::Image2D tmp_project1;
	cl::Image2D tmp_project2;
	cl::Buffer buffer_u;
	cl::Buffer buffer_v;
	cl::Image2D u_in;
	cl::Image2D u_out;
	cl::Image2D v_in;
	cl::Image2D v_out;
	cl::Image2D image;
};

#endif
