#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <ctime>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

vector<string> stationVData;
vector<int> yearVData, monthVData, dayVData, timeVData, tempVData;
string stationName;
float year2, month2, day2, time2, temp2;

void readFile() {
	ifstream theFile; // reading in the file.
	//theFile.open("temp_lincolnshire_short.txt"); // opening the file.
	theFile.open("temp_lincolnshire.txt");
	if (theFile.is_open()) {
		cout << "loading..." << endl;
		// when each data set is pushed into the related variable
		while (theFile >> stationName >> year2 >> month2 >> day2 >> time2 >> temp2) {
			// Push each of them into the specified vector.
			stationVData.push_back(stationName);
			yearVData.push_back(year2);
			monthVData.push_back(month2);
			dayVData.push_back(day2);
			timeVData.push_back(time2);
			tempVData.push_back(temp2 * 100);
		}

		theFile.close();
	}

	else {
		cout << "error reading file" << endl;

	}
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

//	cl_device_id device;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}


	//detect any potential exceptions
	try {

		//print_help();
		//system("pause");
		
		clock_t timerStart = clock();
		readFile();

		// creating an integer array which will contain and convert the float temperature data. this is then reconverted later on.
		vector<int> A(tempVData); 
		//Part 2 - host operations

		// Although AMD, Most Commands were helpful for CUDA.
		// http://amd-dev.wpengine.netdna-cdn.com/wordpress/media/2013/01/Introduction_to_OpenCL_Programming-201005.pdf

		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		
		// get device for queue.
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels_3.cl");

		cl::Program program(context, sources);


		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef int mytype;

		//Part 3 - memory allocation
		//host - input

		//cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
		//int maxGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		//cout << "the max group size:  " << maxGroupSize << endl;

		size_t local_size = 256; // used to define the workgroup sizes. // size_t unsigned __int64
		// output value chan1es if the local_size changes.
		size_t padding_size = A.size() % local_size; // the difference between the temperature size and the work-groups.

		// if the padding_size is instantiated
		if (padding_size) { // padding-size orignally 300...
			vector<int> inputDataVector_ext(local_size - padding_size, 0); // value -- added to the size of A, to make it divisable by the local_size/workgroup sizes.
			A.insert(A.end(), inputDataVector_ext.begin(), inputDataVector_ext.end());
		}
		// this increases the size of the vector by adding padding. this now allows for the work-group sizes to now be divisible by this now padded vector.

		size_t input_elements = A.size();// the amount of elements that are to be pushed into the memory.

		int elements = A.size(); // pushed into the variance vector.

		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		size_t output_size = A.size() * sizeof(mytype);//size in bytes.

		//host - output
		std::vector<mytype> bOutputMean(output_size); // used to store the output from the kernel.
		std::vector<mytype> bOutputMax(output_size);
		std::vector<mytype> bOutputMin(output_size);
		std::vector<mytype> bOutputSD(output_size);


		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size); // contains the size which will hold the original temperature data.
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size); // Outputs the minimum value from the temperature data.
		cl::Buffer buffer_Maximum(context, CL_MEM_READ_WRITE, output_size); // Outputs the maximum value from the temperature data.
		cl::Buffer buffer_Min(context, CL_MEM_READ_WRITE, output_size); // Outputs the maximum value from the temperature data.
		cl::Buffer buffer_Mean(context, CL_MEM_READ_WRITE, output_size); // Outputs the mean value from the temperature data.
		cl::Buffer buffer_Median(context, CL_MEM_READ_WRITE, output_size); // outputs the median value from the temperature data.
		cl::Buffer buffer_SD(context, CL_MEM_READ_WRITE, output_size); // outputs the standard deviation from the temperature data.

		//Part 4 - device operations
		//4.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]); // write buffer, to get the temperatures through.

		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_Min, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_Maximum, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_Mean, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_SD, 0, 0, output_size);//zero B buffer on device memory


		cl::Event prof_event, prof_eventSD, prof_eventmin, prof_eventmax;


		//4.2 Setup and execute all kernels (i.e. device code)

		// ------------- Minimum -------------
		cl::Kernel kernelReduMin = cl::Kernel(program, "minimum");
		kernelReduMin.setArg(0, buffer_A); // reading in the input
		kernelReduMin.setArg(1, buffer_Min); // reading in the output size.
		kernelReduMin.setArg(2, cl::Local(local_size * sizeof(mytype))); // generating local size space. - quicker via cache instead of global memory.
		queue.enqueueNDRangeKernel(kernelReduMin, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_eventmin); // executes the kernel on the device.
		queue.enqueueReadBuffer(buffer_Min, CL_TRUE, 0, output_size, &bOutputMin[0]); // reads from the buffer object to the host memory, so the data can now be accessed on the host.
																					  // bOutput - vector which contain the outputting values.
		cl_ulong profMin = (prof_eventmin.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventmin.getProfilingInfo<CL_PROFILING_COMMAND_START>()); 
		// ^ timer for how long the execution of the kernel will take.

		//4.3 Copy the result from device to host

		// calculation of Minimum.
		float resultMin = (float) bOutputMin[0] / 100; // the first outputting value from the vector, equal to the float variable.
		// exponential division to get the right float value.

		std::cout << "Min Temperature: " << resultMin << " : " << profMin << " ns\n" << std::endl;
		// output of minimum value, including the timing for how long the execution time took in nanoseconds.

		// ------------- Maximum -------------
		cl::Kernel kernelReduMax = cl::Kernel(program, "reduce_Maximum");
		kernelReduMax.setArg(0, buffer_A); // reading in the input
		kernelReduMax.setArg(1, buffer_Maximum); // reading in the output space
		kernelReduMax.setArg(2, cl::Local(local_size * sizeof(mytype))); // generating local size space. - quicker via cache instead of global memory.
		queue.enqueueNDRangeKernel(kernelReduMax, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_eventmax); // executes the kernel on the device.
		queue.enqueueReadBuffer(buffer_Maximum, CL_TRUE, 0, output_size, &bOutputMax[0]); // reads from the buffer object to the host memory, so the data can now be accessed on the host.
		cl_ulong profMax = (prof_eventmax.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventmax.getProfilingInfo<CL_PROFILING_COMMAND_START>());
		// ^ timer for how long the execution of the kernel will take.
		
		// finding/converting the max value to float.
		float resultmaxvalue = (float)bOutputMax[0] / 100;
		std::cout << "Max Temperature: " << resultmaxvalue << " Profiling: " << profMax << " ns\n" << std::endl;

		// ------------- Mean -------------
		cl::Kernel kernelReduMean = cl::Kernel(program, "reduce_add_4");
		kernelReduMean.setArg(0, buffer_A); // reading in the input
		kernelReduMean.setArg(1, buffer_Mean); // reading in the output size.
		kernelReduMean.setArg(2, cl::Local(local_size * sizeof(mytype))); // local_size interpreted in the size per byte of an integer?
		queue.enqueueNDRangeKernel(kernelReduMean, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		queue.enqueueReadBuffer(buffer_Mean, CL_TRUE, 0, output_size, &bOutputMean[0]); // output is then assigned to the bOutput vector.

		cl_ulong prof = (prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

		float resultMeanCalculation = (float)bOutputMean[0] / tempVData.size();
		// calling the size of the vector directly reduces memory leaks for more accurate value.
		float resultMeanValue = resultMeanCalculation / 100; // exponent

		std::cout << "Mean Temperature: " << resultMeanValue << " Profiling: " << prof << " ns\n" << std::endl;

		// Standard Deviation - Begins with Variance
		// 1. starts with the mean.
		// 2. subtract the mean from each value. (if mean is five, take away from each value...)
		// 3. square each difference (each value)
		// 4. calculate the mean of these squared differences.
		// 5. Take the square root of the mean value. sqrt(variable);

		// ------------- Variance -------------
		cl::Kernel kernelVariance = cl::Kernel(program, "get_StandardDeviation");
		kernelVariance.setArg(0, buffer_A); // inputting values.
		kernelVariance.setArg(1, buffer_SD); // outputting value(s).
		kernelVariance.setArg(2, resultMeanValue); // passing in the mean value to the kernel parameters.
	    kernelVariance.setArg(3, elements); // amount of input elements put into an integer the input temperature vector.
		queue.enqueueNDRangeKernel(kernelVariance, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_eventSD); // executes the kernel on the device.
		queue.enqueueReadBuffer(buffer_SD, CL_TRUE, 0, output_size, &bOutputSD[0]); // reading the output from the function call in kernel.

		// calculating the Nano seconds for the time it takes to calculate the functions.
		cl_ulong profVar = (prof_eventSD.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventSD.getProfilingInfo<CL_PROFILING_COMMAND_START>());

		float resultSD = (float)sqrt(bOutputSD[0]) / 100; // Variance found from output then sqrt is used to find standard deviation.
		// division of 100 is taken off as the original float is timed by 100 earlier on so the decimal place is moved forward. 
		//On output it is then moved back when converted back from int to float.
		
		clock_t timerStop = clock();
		float takenTime = ((float)(timerStop - timerStart) / 1000);

		// output to console.
		std::cout << "Standard Deviation Temperature: " << resultSD << " Profiling: " << profVar << " ns\n" << std::endl;
		cout << "Total time taken: " << takenTime << " seconds" << endl;

		system("pause");

	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
