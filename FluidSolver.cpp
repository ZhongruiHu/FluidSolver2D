#include "FluidSolver.h"
#include "Config.h"

constexpr int MEM_SIZE = WIDTH*HEIGHT;

using namespace std;

FluidSolver::FluidSolver()
{
	origin[0] = 0; origin[1] = 0; origin[3] = 0;
	region[0] = WIDTH;
	region[1] = HEIGHT;
	region[2] = 1;
	regionf[0] = region[0] * sizeof(float);
	regionf[1] = region[1] * sizeof(float);
	regionf[2] = 1;

	origin_work = cl::NDRange(0, 0);
	region_work = cl::NDRange(WIDTH, HEIGHT);

	origin_work_center = cl::NDRange(1, 1);
	region_work_center = cl::NDRange(WIDTH - 2, HEIGHT - 2);
}

FluidSolver::~FluidSolver()
{
	queue.finish();
}

void FluidSolver::initialization()
{
	cl_init();
	program_init();
}

void FluidSolver::cl_init() 
{
	// opencl init
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0) {
		cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	auto id_platform = PLATFORM;
	if (id_platform >= all_platforms.size()) {
		cout << " Warning: Default platform used (Wrong configuration)\n";
		id_platform = 0;
	}
	default_platform = all_platforms[PLATFORM];
	cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
	//get default device of the default platform
	vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device = all_devices[0];
	cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	context = cl::Context({ default_device });
	queue = cl::CommandQueue(context, default_device);

	// load opencl source
	ifstream cl_file("core.cl");
	string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
	cl::Program::Sources source(1, make_pair(cl_string.c_str(), cl_string.length() + 1));

	// create program
	program = cl::Program(context, source);
	if (program.build({ default_device }) != CL_SUCCESS) {
		cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		exit(1);
	} else {
		cout << "Build sucessful" << endl;
	}
}

void FluidSolver::program_init() {
	static const cl::ImageFormat format_float1 = { CL_R, CL_FLOAT };
	kernel_diffuse   = cl::Kernel(program, "diffuse");
	kernel_advect    = cl::Kernel(program, "advect");
	kernel_project1  = cl::Kernel(program, "project1");
	kernel_project2  = cl::Kernel(program, "project2");
	kernel_draw_img  = cl::Kernel(program, "floatToR");
	kernel_reset     = cl::Kernel(program, "reset");
	kernel_addsource = cl::Kernel(program, "addCircleValue");

	density_in =	cl::Image2D(context, CL_MEM_READ_WRITE, format_float1, WIDTH, HEIGHT, 0);
	density_out =	cl::Image2D(context, CL_MEM_READ_WRITE, format_float1, WIDTH, HEIGHT, 0);
	u_in =			cl::Image2D(context, CL_MEM_READ_WRITE, format_float1, WIDTH, HEIGHT, 0);
	v_in =			cl::Image2D(context, CL_MEM_READ_WRITE, format_float1, WIDTH, HEIGHT, 0);
	u_out =			cl::Image2D(context, CL_MEM_READ_WRITE, format_float1, WIDTH, HEIGHT, 0);
	v_out =			cl::Image2D(context, CL_MEM_READ_WRITE, format_float1, WIDTH, HEIGHT, 0);
	image =			cl::Image2D(context, CL_MEM_READ_WRITE, { CL_RGBA, CL_UNSIGNED_INT8 }, WIDTH, HEIGHT, 0);
	tmp_project1 =	cl::Image2D(context, CL_MEM_READ_WRITE, format_float1, WIDTH, HEIGHT, 0);
	tmp_project2 =	cl::Image2D(context, CL_MEM_READ_WRITE, format_float1, WIDTH, HEIGHT, 0);
	buffer_u =		cl::Buffer(context,  CL_MEM_READ_WRITE, WIDTH*HEIGHT * sizeof(float));
	buffer_v =		cl::Buffer(context,  CL_MEM_READ_WRITE, WIDTH*HEIGHT * sizeof(float));
}


void FluidSolver::add_source(cl::Image2D& in_out,int x, int y, int radius, float intensity)
{
	kernel_addsource.setArg(0, in_out);
	kernel_addsource.setArg(1, in_out);
	kernel_addsource.setArg(2, x);
	kernel_addsource.setArg(3, y);
	kernel_addsource.setArg(4, intensity);
	kernel_addsource.setArg(5, (float)radius - 0.5f);
	const int bound_width = (x + radius < WIDTH-1) ? 2 * radius : (WIDTH) - (x - radius);
	const int bound_height = (y + radius < HEIGHT-1) ? 2 * radius : (HEIGHT) - (y - radius);
	const int bound_top = (x - radius < 1) ? 1 : x - radius;
	const int bound_left = (y - radius < 1) ? 1 : y - radius;
	queue.enqueueNDRangeKernel(kernel_addsource, cl::NDRange(bound_top, bound_left), cl::NDRange(bound_width, bound_height), cl::NullRange);
}

void FluidSolver::add_pressure(int x, int y, int radius, float intensity)
{
	add_source(density_in, x, y, radius, intensity);
}

void FluidSolver::add_velocity(int x, int y, float dx, float dy, float force, int radius)
{
	add_source(u_in, x, y, radius, dx*force);
	add_source(v_in, x, y, radius, dy*force);
}

void FluidSolver::set_data_image(cl_uint8 * img)
{
	data_image = img;
}

void FluidSolver::update_image()
{
	kernel_draw_img.setArg(0, density_in);
	kernel_draw_img.setArg(1, image);
	queue.enqueueNDRangeKernel(kernel_draw_img, origin_work, region_work, cl::NullRange);
	queue.enqueueReadImage(image, CL_TRUE, origin, region, 0, 0, data_image);
}

void FluidSolver::reset()
{
	cl::Image2D* images[] = { &density_in, &density_out, &u_in, &u_out, &v_in, &v_out, &image };
	for (int i = 0; i < 7;++i) {
		kernel_reset.setArg(0, *images[i]);
		queue.enqueueNDRangeKernel(kernel_reset, cl::NDRange(0, 0), cl::NDRange(WIDTH, HEIGHT), cl::NullRange);
	}
}

void FluidSolver::update(float dt)
{
	constexpr auto VISCO_DIV = 1.0f + 4.0f*VISCO;
	if (dt > 0.02f) { // clamp update rate else the error is too high
		dt = 0.02f;
	}
	const float a = dt*DIFF_DENSITY*WIDTH*HEIGHT;
	// velocity -----------------------
	diffuse(u_out, u_in, VISCO, VISCO_DIV, 1);
	diffuse(v_out, v_in, VISCO, VISCO_DIV, 2);

	project(u_out, v_out);
	queue.enqueueCopyBufferToImage(buffer_u, u_out, 0, origin, region);
	queue.enqueueCopyBufferToImage(buffer_v, v_out, 0, origin, region);
	

	advect(u_in, u_out, u_in, v_in, dt, 1);
	advect(v_in, v_out, u_in, v_in, dt, 2);

	queue.enqueueCopyImage(u_in, u_out, origin, origin, region);
	queue.enqueueCopyImage(v_in, v_out, origin, origin, region);
	project(u_out, v_out);
	queue.enqueueCopyBufferToImage(buffer_u, u_in, 0, origin, region);
	queue.enqueueCopyBufferToImage(buffer_v, v_in, 0, origin, region);

	// density ------------------------
	diffuse(density_out, density_in, a, 1 + 4.0f*a, 0);
	advect(density_in, density_out, u_in, v_in, dt, 0);

}


inline void FluidSolver::diffuse(cl::Image2D & input_output, const cl::Image2D & src, float diff, float diff_div, int bound) {
	if (diff_div == 0.0f) diff_div = 0.000000000001f;
	kernel_diffuse.setArg(0, input_output);
	kernel_diffuse.setArg(1, input_output);
	kernel_diffuse.setArg(2, src);
	kernel_diffuse.setArg(3, diff);
	kernel_diffuse.setArg(4, diff_div);
	for (unsigned int k = 0; k < SOLVER_NB_ITERATIONS; ++k) {
		queue.enqueueNDRangeKernel(kernel_diffuse, origin_work, region_work, cl::NullRange);
	}
}

inline void FluidSolver::advect(cl::Image2D & dest, const cl::Image2D & src, cl::Image2D & img_u, cl::Image2D & img_v, float dt, int bound)
{
	kernel_advect.setArg(0, src);
	kernel_advect.setArg(1, dest);
	kernel_advect.setArg(2, img_u);
	kernel_advect.setArg(3, img_v);
	kernel_advect.setArg(4, dt);
	kernel_advect.setArg(5, WIDTH);
	kernel_advect.setArg(6, HEIGHT);
	queue.enqueueNDRangeKernel(kernel_advect, origin_work_center, region_work_center, cl::NullRange);
}

inline void FluidSolver::project(cl::Image2D & img_u, cl::Image2D & img_v)
{
	constexpr float hx = 1.0f / WIDTH, hy = 1.0f / HEIGHT;

	kernel_project1.setArg(0, tmp_project1);
	kernel_project1.setArg(1, img_u);
	kernel_project1.setArg(2, img_v);
	kernel_project1.setArg(3, hx);
	kernel_project1.setArg(4, hy);
	queue.enqueueNDRangeKernel(kernel_project1, origin_work_center, region_work_center, cl::NullRange);

	kernel_reset.setArg(0, tmp_project2);
	queue.enqueueNDRangeKernel(kernel_reset, cl::NDRange(0, 0), cl::NDRange(WIDTH, HEIGHT), cl::NullRange);
	diffuse(tmp_project2, tmp_project1, 1.0f, 4.0f, 0);

	kernel_project2.setArg(0, tmp_project2);
	kernel_project2.setArg(1, buffer_u);
	kernel_project2.setArg(2, buffer_v);
	kernel_project2.setArg(3, WIDTH);
	kernel_project2.setArg(4, HEIGHT);
	queue.enqueueCopyImageToBuffer(img_u, buffer_u, origin, region, 0);
	queue.enqueueCopyImageToBuffer(img_v, buffer_v, origin, region, 0);
	queue.enqueueNDRangeKernel(kernel_project2, origin_work_center, region_work_center, cl::NullRange);
}