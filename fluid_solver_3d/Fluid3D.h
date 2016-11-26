#ifndef FLUID3D_H
#define FLUID3D_H

#include <fstream>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class Fluid3D
{
public:
	Fluid3D(cl::Context ctx, cl::Device dev);
	Fluid3D(cl::Context context, cl::Device device, unsigned int width, unsigned int height, unsigned int depth);
	virtual ~Fluid3D();
	void setSize(unsigned int width, unsigned int height, unsigned int depth);
	/** Return true if complete sucess */
	bool initialization();
	void update(float dt);
	void updateImage();
	void setDataImage(cl_uint8 * img);
	unsigned int getWidth() const;
	unsigned int getHeight() const;
	unsigned int getDepth() const;
	void reset();
	void save();
	void addPressure(int posx, int posy, int radius, float pressure);
	void addVelocity(int posx, int posy, int deltax, int deltay, float intensity, int radius);
	

private:
	void diffuseDensity(float a, float div);
	void diffuseVelocity();
	void project(cl::Buffer & src, cl::Buffer & dest);
	void project2();
	void advectVelocity();
	void advectDensity();
	void project1();
	void project();
	void exportDf3();

	unsigned int width;
	unsigned int height;
	unsigned int depth;
	unsigned int volume;
	
	float visco;
	float visco_div;
	float density_factor;
	//
	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
	// region work
	cl::size_t<3> origin;
	cl::size_t<3> region;
	cl::size_t<3> regionf;
	cl::NDRange origin_work;
	cl::NDRange region_work;
	cl::size_t<3> origin2d;
	cl::size_t<3> region2d;
	cl::NDRange origin_work2d;
	cl::NDRange region_work2d;
	cl::NDRange origin_work_center;
	cl::NDRange region_work_center;
	// opencl kernels
	cl::Kernel kernel_diffuse;
	cl::Kernel kernel_diffuse_tmp;
	cl::Kernel kernel_diffuse_v;
	cl::Kernel kernel_advect_density;
	cl::Kernel kernel_advect_velocity;
	cl::Kernel kernel_project1;
	cl::Kernel kernel_project2;
	cl::Kernel kernel_project1bis;
	cl::Kernel kernel_project2bis;
	cl::Kernel kernel_reset_buffer;
	cl::Kernel kernel_reset_buffer3D;
	cl::Kernel kernel_addsource;
	cl::Kernel kernel_addsource3D;
	cl::Kernel kernel_draw_img;
	// gpu memory structures
	cl_uint8* data_image;// pointer on the sfml image memory
	cl::Image2D image;
	cl::Buffer  density;
	cl::Buffer  density2;
	cl::Buffer velocity;
	cl::Buffer velocity2;
	cl::Buffer tmp_project;
	cl::Buffer tmp_project2;

	int count = 0;
	int t;
	bool isSaving = false;
};

#endif
