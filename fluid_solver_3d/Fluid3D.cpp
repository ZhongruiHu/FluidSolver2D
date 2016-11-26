#include "Fluid3D.h"

#include "D3fWriter.hpp"
#include "config.hpp"
#include <cstdint>
#include <iostream>
#include <thread>

using namespace std;

static std::string getErrorStr(cl_int error);

static constexpr float VISCO = 0.00001f;
static constexpr float VISCO_DIV = 1.0f + 6.0f*VISCO;
static constexpr float DIFF_DENSITY = 0.000001f;
static constexpr unsigned int SOLVER_NB_ITERATIONS = 16;

Fluid3D::Fluid3D(cl::Context ctx, cl::Device dev) : 
		context(ctx), device(dev)
{
	width = DEFAULT_WIDTH;
	height = DEFAULT_HEIGHT;
	depth = DEFAULT_DEPTH;
	volume = width*height*depth;

	density_factor = DIFF_DENSITY*volume;
}

Fluid3D::Fluid3D(cl::Context ctx, cl::Device dev, unsigned int w, unsigned int h, unsigned int d) : 
	context(ctx), device(dev), width(w), height(h), depth(d)
{
	volume = width*height*depth;

	density_factor = DIFF_DENSITY*volume;
}

Fluid3D::~Fluid3D()
{
	queue.finish();
}

void Fluid3D::setDataImage(cl_uint8 * img)
{
	data_image = img;
}

void Fluid3D::updateImage()
{
	queue.enqueueNDRangeKernel(kernel_draw_img, origin_work2d, region_work2d, cl::NullRange);
	queue.enqueueReadImage(image, CL_TRUE, origin2d, region2d, 0, 0, data_image);
}

bool Fluid3D::initialization()
{
	//cout << "Init" << endl;
	queue = cl::CommandQueue(context, device);
	
	// load opencl source
	ifstream cl_file("../core.cl");
	if (!cl_file.good())
	{
		cout << "core.cl not found" << endl;
	}
	string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
	cl::Program::Sources source(1, make_pair(cl_string.c_str(), cl_string.length() + 1));

	// create program and build it
	program = cl::Program(context, source);
	if (program.build({ device }) != CL_SUCCESS) {
		cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		return false;
	} else {
		cout << "Build sucessful" << endl;
	}

	// ----
	// init work region and origin

	origin[0] = 0; origin[1] = 0; origin[3] = 0;
	region[0] = width;
	region[1] = height;
	region[2] = depth;
	regionf[0] = region[0] * sizeof(float);
	regionf[1] = region[1] * sizeof(float);
	regionf[2] = region[2] * sizeof(float);

	origin_work = cl::NDRange(0, 0, 0);
	region_work = cl::NDRange(width, height, depth);

	origin2d[0] = 0; origin2d[1] = 0; origin2d[3] = 0;
	region2d[0] = region[0];
	region2d[1] = region[1];
	region2d[2] = 1;
	origin_work2d = cl::NDRange(0, 0);
	region_work2d = cl::NDRange(width, height);

	origin_work_center = cl::NDRange(1, 1, 1);
	region_work_center = cl::NDRange(width - 2, height - 2, depth - 2);

	// --
	// Initialize memory
	image =			cl::Image2D(context, CL_MEM_READ_WRITE, { CL_RGBA, CL_UNSIGNED_INT8 }, width, height, 0);
	
	density =		cl::Buffer(context, CL_MEM_READ_WRITE, volume * sizeof(float));
	density2 =		cl::Buffer(context, CL_MEM_READ_WRITE, volume * sizeof(float));
	velocity =		cl::Buffer(context, CL_MEM_READ_WRITE, volume * 3 * sizeof(float));
	velocity2 =		cl::Buffer(context, CL_MEM_READ_WRITE, volume * 3 * sizeof(float));
	tmp_project  =	cl::Buffer(context, CL_MEM_READ_WRITE, volume * sizeof(float));
	tmp_project2 =	cl::Buffer(context, CL_MEM_READ_WRITE, volume * sizeof(float));

	// Create the kernels
	kernel_diffuse   = cl::Kernel(program, "diffuse");
	kernel_diffuse.setArg(0, density2);
	kernel_diffuse.setArg(1, density);
	kernel_diffuse.setArg(4, width);
	kernel_diffuse.setArg(5, height);
	kernel_diffuse.setArg(6, depth);

	kernel_advect_density = cl::Kernel(program, "advect");
	kernel_advect_density.setArg(0,density);
	kernel_advect_density.setArg(1,density2);
	kernel_advect_density.setArg(2,velocity);
	kernel_advect_density.setArg(3,width);
	kernel_advect_density.setArg(4,height);
	kernel_advect_density.setArg(5,depth);
	kernel_advect_density.setArg(6,0.02f);

	kernel_draw_img  = cl::Kernel(program, "drawScreen");
	kernel_draw_img.setArg(0, density);
	kernel_draw_img.setArg(1, image);
	kernel_draw_img.setArg(2, width);
	kernel_draw_img.setArg(3, height);

	kernel_addsource = cl::Kernel(program, "addSource");
	kernel_addsource.setArg(0, density);
	kernel_addsource.setArg(6, width);
	kernel_addsource.setArg(7, height);

	kernel_addsource3D = cl::Kernel(program, "addSource3D");
	kernel_addsource3D.setArg(0, velocity);
	kernel_addsource3D.setArg(7, width);
	kernel_addsource3D.setArg(8, height);

	kernel_reset_buffer = cl::Kernel(program, "resetBuffer");
	kernel_reset_buffer.setArg(1, width);
	kernel_reset_buffer.setArg(2, height);

	kernel_reset_buffer3D = cl::Kernel(program, "resetBuffer3D");
	kernel_reset_buffer3D.setArg(1, width);
	kernel_reset_buffer3D.setArg(2, height);

	//---
	kernel_diffuse_v = cl::Kernel(program, "diffuse3D");

	kernel_diffuse_v.setArg(0, velocity2);
	kernel_diffuse_v.setArg(1, velocity);
	kernel_diffuse_v.setArg(2, VISCO);
	kernel_diffuse_v.setArg(3, VISCO_DIV);
	kernel_diffuse_v.setArg(4, width);
	kernel_diffuse_v.setArg(5, height);
	kernel_diffuse_v.setArg(6, depth);

	kernel_advect_velocity = cl::Kernel(program, "advect3D");
	kernel_advect_velocity.setArg(0,velocity);
	kernel_advect_velocity.setArg(1,velocity2);
	kernel_advect_velocity.setArg(2,width);
	kernel_advect_velocity.setArg(3,height);
	kernel_advect_velocity.setArg(4,depth);
	kernel_advect_velocity.setArg(5,0.02f);

	kernel_project1 = cl::Kernel(program, "project1");
	kernel_project1.setArg(0, tmp_project);
	kernel_project1.setArg(1, velocity2);
	kernel_project1.setArg(2, width);
	kernel_project1.setArg(3, height);
	kernel_project1.setArg(4, depth);

	kernel_project2 = cl::Kernel(program, "project2");
	kernel_project2.setArg(0, tmp_project2);
	kernel_project2.setArg(1, velocity2);
	kernel_project2.setArg(2, width);
	kernel_project2.setArg(3, height);
	kernel_project2.setArg(4, depth);

	kernel_project1bis = cl::Kernel(program, "project1");
	kernel_project1bis.setArg(0, tmp_project);
	kernel_project1bis.setArg(1, velocity);
	kernel_project1bis.setArg(2, width);
	kernel_project1bis.setArg(3, height);
	kernel_project1bis.setArg(4, depth);

	kernel_project2bis = cl::Kernel(program, "project2");
	kernel_project2bis.setArg(0, tmp_project2);
	kernel_project2bis.setArg(1, velocity);
	kernel_project2bis.setArg(2, width);
	kernel_project2bis.setArg(3, height);
	kernel_project2bis.setArg(4, depth);

	kernel_diffuse_tmp = cl::Kernel(program, "diffuse");// diffuse tmp
	kernel_diffuse_tmp.setArg(0, tmp_project2);
	kernel_diffuse_tmp.setArg(1, tmp_project);
	kernel_diffuse_tmp.setArg(2, 1.0f);
	kernel_diffuse_tmp.setArg(3, 6.0f);
	kernel_diffuse_tmp.setArg(4, width);
	kernel_diffuse_tmp.setArg(5, height);
	kernel_diffuse_tmp.setArg(6, depth);

	// reset all buffers to zero
	reset();

	return true;
}

void Fluid3D::update(float dtt)
{
	const float dt = (dtt < 0.02f) ? dtt : 0.02f;
	const float a = dt*density_factor;
	// velocity step ------------------
	diffuseVelocity();
	project1();
	advectVelocity();
	project2();
	// density step -------------------
	diffuseDensity(a, 1 + 6.0f*a);
	advectDensity();

	if (isSaving) {
		++t;
		exportDf3();
	}
}

void Fluid3D::diffuseDensity(float a, float div)
{
	kernel_diffuse.setArg(2, a);
	kernel_diffuse.setArg(3, div);
	for (unsigned int k = 0; k < SOLVER_NB_ITERATIONS; ++k) {
		queue.enqueueNDRangeKernel(kernel_diffuse, origin_work_center, region_work_center, cl::NullRange);
	}
}

void Fluid3D::diffuseVelocity()
{

	for (unsigned int k = 0; k < SOLVER_NB_ITERATIONS; ++k) {
		queue.enqueueNDRangeKernel(kernel_diffuse_v, origin_work_center, region_work_center, cl::NullRange);
	}
}

void Fluid3D::advectDensity()
{
	queue.enqueueNDRangeKernel(kernel_advect_density, origin_work_center, region_work_center, cl::NullRange);
}

void Fluid3D::project1()
{
	queue.enqueueNDRangeKernel(kernel_project1, origin_work_center, region_work_center, cl::NullRange);

	kernel_reset_buffer.setArg(0, tmp_project2);
	queue.enqueueNDRangeKernel(kernel_reset_buffer, origin_work, region_work, cl::NullRange);
	for (unsigned int k = 0; k < SOLVER_NB_ITERATIONS; ++k) {
		queue.enqueueNDRangeKernel(kernel_diffuse_tmp, origin_work_center, region_work_center, cl::NullRange);
	}
	queue.enqueueNDRangeKernel(kernel_project2, origin_work_center, region_work_center, cl::NullRange);
}

void Fluid3D::project2()
{
	queue.enqueueNDRangeKernel(kernel_project1bis, origin_work_center, region_work_center, cl::NullRange);

	kernel_reset_buffer.setArg(0, tmp_project2);
	queue.enqueueNDRangeKernel(kernel_reset_buffer, origin_work, region_work, cl::NullRange);
	for (unsigned int k = 0; k < SOLVER_NB_ITERATIONS; ++k) {
		queue.enqueueNDRangeKernel(kernel_diffuse_tmp, origin_work_center, region_work_center, cl::NullRange);
	}
	queue.enqueueNDRangeKernel(kernel_project2bis, origin_work_center, region_work_center, cl::NullRange);
}

void Fluid3D::advectVelocity()
{
	queue.enqueueNDRangeKernel(kernel_advect_velocity, origin_work_center, region_work_center, cl::NullRange);
}

void Fluid3D::addPressure(int x, int y, int radius, float pressure)
{
	int z = depth/2;

	kernel_addsource.setArg(1, x);
	kernel_addsource.setArg(2, y);
	kernel_addsource.setArg(3, z);
	kernel_addsource.setArg(4, pressure);
	kernel_addsource.setArg(5, (float)radius - 0.5f);

	const int bound_width  = (x + radius+1 < (int)width)  ? 2 * radius : (width-2)  - (x - radius);
	const int bound_height = (y + radius+1 < (int)height) ? 2 * radius : (height-2) - (y - radius);
	int bound_depth        = (z + radius+1 < (int)depth)  ? 2 * radius : (depth-2) - (z - radius);
	bound_depth = (bound_depth+2 < (int)depth ) ? bound_depth : (int)depth-2;
	const int bound_top  = (x - radius < 1) ? 1 : x - radius;
	const int bound_left = (y - radius < 1) ? 1 : y - radius;
	const int bound_up   = (z - radius < 1) ? 1 : z - radius;
	queue.enqueueNDRangeKernel(kernel_addsource, cl::NDRange(bound_top, bound_left, bound_up), cl::NDRange(bound_width, bound_height, bound_depth), cl::NullRange);
}

void Fluid3D::addVelocity(int x, int y, int deltax, int deltay, float intensity, int radius)
{
	int z = depth/2;
	
	kernel_addsource3D.setArg(1, x);
	kernel_addsource3D.setArg(2, y);
	kernel_addsource3D.setArg(3, z);
	kernel_addsource3D.setArg(4, deltax*intensity);
	kernel_addsource3D.setArg(5, deltay*intensity);
	kernel_addsource3D.setArg(6, (float)radius - 0.5f);

	const int bound_width  = (x + radius+1 < (int)width)  ? 2 * radius : (width-2)  - (x - radius);
	const int bound_height = (y + radius+1 < (int)height) ? 2 * radius : (height-2) - (y - radius);
	int bound_depth        = (z + radius+1 < (int)depth)  ? 2 * radius : (depth-2) - (z - radius);
	bound_depth = (bound_depth+2 < (int)depth ) ? bound_depth : (int)depth-2;
	const int bound_top  = (x - radius < 1) ? 1 : x - radius;
	const int bound_left = (y - radius < 1) ? 1 : y - radius;
	const int bound_up   = 1;//(z - radius < 1) ? 1 : z - radius;
	queue.enqueueNDRangeKernel(kernel_addsource3D, cl::NDRange(bound_top, bound_left, bound_up), cl::NDRange(bound_width, bound_height, bound_depth), cl::NullRange);
}

void Fluid3D::save()
{
	isSaving = !isSaving;
	count = 0;
}

void Fluid3D::exportDf3()
{
	float* data = new float[volume];
	queue.enqueueReadBuffer(density,CL_TRUE,0,volume*sizeof(float),data);
	// data will be deleted by the thread
	std::thread thread(&D3fWriter::exportdf3,"render"+std::to_string(count)+".df3",data, width,height, depth);
	thread.detach();
	++count;
}

void Fluid3D::reset()
{

	{
		cl::Buffer* data[] = { &density, &density2 };
		for (auto & buffer : data) {
			kernel_reset_buffer.setArg(0, *buffer);
			queue.enqueueNDRangeKernel(kernel_reset_buffer,origin_work, region_work, cl::NullRange);
		}
	}
	{
		cl::Buffer* data[] = { &velocity, &velocity2 };
		for (auto & buffer : data) {
			kernel_reset_buffer3D.setArg(0, *buffer);
			queue.enqueueNDRangeKernel(kernel_reset_buffer3D, origin_work, region_work, cl::NullRange);
		}
	}
	count = 0;
}

unsigned int Fluid3D::getWidth() const
{
	return width;
}
unsigned int Fluid3D::getHeight() const
{
	return height;
}
unsigned int Fluid3D::getDepth() const
{
	return depth;
}
