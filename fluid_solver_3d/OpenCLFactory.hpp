#ifndef OPENCL_FACTORY_H
#define OPENCL_FACTORY_H

#include "config.hpp"

#include <fstream>
#include <vector>
#include <iostream>
#include <CL/cl.hpp>

namespace OpenCLFactory
{

	/** Create an opencl context and give the corresponding device */
	std::pair<cl::Device,cl::Context> createContext()
	{
		using namespace std;
		vector<cl::Platform> all_platforms;
		
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
		auto default_platform = all_platforms[PLATFORM];
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

		auto context = cl::Context({ default_device });
		
		cout << "Context made"<<endl; 
		return pair<cl::Device,cl::Context>(default_device,context);
	}


	std::string getErrorStr(cl_int error)
	{
		switch(error){
		case CL_SUCCESS: return "Success";
		case CL_DEVICE_NOT_FOUND: return "Device Not Found";
		case CL_DEVICE_NOT_AVAILABLE: return "Device Not Available";
		case CL_COMPILER_NOT_AVAILABLE: return "Compiler Not Available";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory Object Allocation Failure";
		case CL_OUT_OF_RESOURCES: return "Out of Resources";
		case CL_OUT_OF_HOST_MEMORY: return "Out of Host Memory";
		case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling Information Not Available";
		case CL_MEM_COPY_OVERLAP: return "Memory Copy Overlap";
		case CL_IMAGE_FORMAT_MISMATCH: return "Image Format Mismatch";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image Format Not Supported";
		case CL_BUILD_PROGRAM_FAILURE: return "Build Program Failure";
		case CL_MAP_FAILURE: return "Map Failure";
		case CL_INVALID_VALUE: return "Invalid Value";
		case CL_INVALID_DEVICE_TYPE: return "Invalid Device Type";
		case CL_INVALID_PLATFORM: return "Invalid Platform";
		case CL_INVALID_DEVICE: return "Invalid Device";
		case CL_INVALID_CONTEXT: return "Invalid Context";
		case CL_INVALID_QUEUE_PROPERTIES: return "Invalid Queue Properties";
		case CL_INVALID_COMMAND_QUEUE: return "Invalid Command Queue";
		case CL_INVALID_HOST_PTR: return "Invalid Host Pointer";
		case CL_INVALID_MEM_OBJECT: return "Invalid Memory Object";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid Image Format Descriptor";
		case CL_INVALID_IMAGE_SIZE: return "Invalid Image Size";
		case CL_INVALID_SAMPLER: return "Invalid Sampler";
		case CL_INVALID_BINARY: return "Invalid Binary";
		case CL_INVALID_BUILD_OPTIONS: return "Invalid Build Options";
		case CL_INVALID_PROGRAM: return "Invalid Program";
		case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid Program Executable";
		case CL_INVALID_KERNEL_NAME: return "Invalid Kernel Name";
		case CL_INVALID_KERNEL_DEFINITION: return "Invalid Kernel Definition";
		case CL_INVALID_KERNEL: return "Invalid Kernel";
		case CL_INVALID_ARG_INDEX: return "Invalid Argument Index";
		case CL_INVALID_ARG_VALUE: return "Invalid Argument Value";
		case CL_INVALID_ARG_SIZE: return "Invalid Argument Size";
		case CL_INVALID_KERNEL_ARGS: return "Invalid Kernel Arguments";
		case CL_INVALID_WORK_DIMENSION: return "Invalid Work Dimension";
		case CL_INVALID_WORK_GROUP_SIZE: return "Invalid Work Group Size";
		case CL_INVALID_WORK_ITEM_SIZE: return "Invalid Work Item Size";
		case CL_INVALID_GLOBAL_OFFSET: return "Invalid Global Offset";
		case CL_INVALID_EVENT_WAIT_LIST: return "Invalid Event Wait List";
		case CL_INVALID_EVENT: return "Invalid Event";
		case CL_INVALID_OPERATION: return "Invalid Operation";
		case CL_INVALID_GL_OBJECT: return "Invalid GL Object";
		case CL_INVALID_BUFFER_SIZE: return "Invalid Buffer Size";
		case CL_INVALID_MIP_LEVEL: return "Invalid MIP Level";
		case CL_INVALID_GLOBAL_WORK_SIZE: return "Invalid Global Work Size";
#ifdef CL_VERSION_1_2
		case CL_COMPILE_PROGRAM_FAILURE: return "Compile Program Failure";
		case CL_LINKER_NOT_AVAILABLE: return "Linker Not Available";
		case CL_LINK_PROGRAM_FAILURE: return "Link Program Failure";
		case CL_DEVICE_PARTITION_FAILED: return "Device Partition Failed";
		case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "Kernel Argument Info Not Available";
		case CL_INVALID_PROPERTY: return "Invalid Property";
		case CL_INVALID_IMAGE_DESCRIPTOR: return "Invalid Image Descriptor";
		case CL_INVALID_COMPILER_OPTIONS: return "Invalid Compiler Options";
		case CL_INVALID_LINKER_OPTIONS: return "Invalid Linker Options";
		case CL_INVALID_DEVICE_PARTITION_COUNT: return "Invalid Device Partition Count";
#endif // CL_VERSION_1_2
#ifdef CL_VERSION_2_0
		case CL_INVALID_PIPE_SIZE: return "Invalid Pipe Size";
		case CL_INVALID_DEVICE_QUEUE: return "Invalid Device Queue";
#endif
		default: return "Unknown OpenCL Error";

		}
	}
}

#endif