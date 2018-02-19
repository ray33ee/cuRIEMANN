#include <stdio.h>

#include <math.h>
#include <new> //std::nothrow

#include "Complex.h"

#include "kernel.h"

int const threadCount = 192; //Number of threads per block
const int MAX_STR_BUFF = 40;

//Dimensions
unsigned globalWidth;
unsigned globalHeight;

//Flags
bool isHost; //host = true, false = device
bool isShared; //shared memory = true, global memory = false

LOG_STRING log_str; //For debugging purposes
bool isForced;

//Arrays
Token* globalList;
RGB* globalDeviceResults;
RGB* globalHostResults;
Complex* globalStack;
Complex* globalDoubleStack;

unsigned globalListCount;
unsigned globalMaxStack;

//Array sizes
unsigned resultsSize;
unsigned stackSize;

__host__ ERRORCODES entryConstruct(bool force_host, LOG_STRING lstr)
{
	int deviceCount;

	log_str = lstr;
	isForced = force_host;

	if (log_str != nullptr)
	{
		log_str("Force Host: ");
		if (force_host)
			log_str("On\n");
		else
			log_str("Off\n");
	}

	if (log_str != nullptr)
		log_str("Entry Construct\n");

	cudaError_t firstCall = cudaGetDeviceCount(&deviceCount); 

	if (isForced) //Force host
		isHost = true;
	else  //Try device, otherwise use host
		isHost = firstCall == cudaErrorNoDevice || firstCall == cudaErrorInsufficientDriver;//No device or no proper device drivers, use host

	if (!isHost)  //Use Device
	{
		int device;
		for (device = 0; device < deviceCount; ++device)
		{
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, device);
			if (log_str != nullptr)
			{
				char* s = new char[MAX_STR_BUFF];
				sprintf(s, "	Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
				log_str(s);
			}
		}

		{
			int* a;
			if (cudaMalloc(&a, sizeof(int)) == cudaErrorMemoryAllocation)
				return Construct_Test_MemAlloc;
			if (cudaFree(a) == cudaErrorInvalidDevicePointer)
				return Construct_Test_InvalidPointer;
		}
		if (log_str != nullptr)
			log_str("	Using device...\n");

	}
	else //Otherwise use host
	{
		if (log_str != nullptr)
			log_str("	Using Host...\n");
	}
	
	

	globalList = nullptr;
	globalDeviceResults = nullptr;
	globalHostResults = nullptr;
	globalStack = nullptr;
	globalDoubleStack = nullptr;

	stackSize = 0;
	resultsSize = 0;
	return Success;
}

__host__ ERRORCODES entryInitialise(unsigned width, unsigned height, TokenList list, RGB* results)
{
	globalMaxStack = list.getStackSize();

	if (log_str != nullptr)
		log_str("Entry Initialise\n");

	if (isHost) //Uses - results, d_stack, globalList
	{
		if (stackSize < width * height * sizeof(Complex) * globalMaxStack) //Current stack array is too small, reallocate
		{
			if (log_str != nullptr)
				log_str("	Reallocate host stack - Host\n");
			//delete[] globalStack; //Delete current stack array if allocated
			globalStack = new (std::nothrow) Complex[width*height*globalMaxStack]; //Construct new stack array
			if (globalStack == nullptr)
				return Initialise_Host_Stack_MemAlloc;
			stackSize = width * height * globalMaxStack * sizeof(Complex); //Update stackSize
		}
		//delete[] globalList; //free current token list
		globalList = list.formula(); //Update new tokenlist
	}
	else //Uses - globalList, d_results, results and d_stack
	{
		if (resultsSize < width * height * sizeof(Complex)) //Current results array is too small, reallocate
		{
			if (log_str != nullptr)
			{
				char* s1 = new char[MAX_STR_BUFF];
				char* s2 = new char[MAX_STR_BUFF];
				sprintf(s1, "	Reallocate device results - Device - old size: %i, new size: %i\n", resultsSize, width * height * sizeof(Complex));
				sprintf(s2, "	width: %i, height: %i, sizeof: %i\n", width, height, sizeof(Complex));
				log_str(s1);
				log_str(s2);
			}
			if (cudaFree(globalDeviceResults) == cudaErrorInvalidDevicePointer) //Delete current device results array if allocated
				return Initialise_Device_DeviceResults_InvalidPointer;
			if (cudaMalloc(&globalDeviceResults, width * height * sizeof(RGB)) == cudaErrorMemoryAllocation) //Construct new device array
				return Initialise_Device_DeviceResults_MemAlloc;
			resultsSize = width * height * sizeof(RGB); //Update resultSize
		}

		if (log_str != nullptr)
		{
			char* s = new char[MAX_STR_BUFF];
			sprintf(s, "	Stack max: %i\n", globalMaxStack);
			log_str(s);
		}
		isShared = globalMaxStack < 5;

		if (!isShared && stackSize < width * height * sizeof(Complex) * globalMaxStack) //If we're using global memory and the memory needs expanding...
		{
			if (log_str != nullptr)
				log_str("	Reallocate device stack - device\n");
			if (cudaFree(globalStack) == cudaErrorInvalidDevicePointer) //Delete current stack array if allocated
				return Initialise_Device_Stack_InvalidPointer;
			if (cudaMalloc(&globalStack, width * height * globalMaxStack * sizeof(Complex)) == cudaErrorMemoryAllocation) //Construct new stack array
				return Initialise_Device_Stack_MemAlloc;
			stackSize = width * height * globalMaxStack * sizeof(Complex); //Update stackSize
		}
		if (log_str != nullptr)
			log_str("	Copying list to device\n");
		if (cudaFree(globalList) == cudaErrorInvalidDevicePointer) //Free device token list
			return Initialise_Device_List_InvalidPointer;
		if (cudaMalloc(&globalList, list.count() * sizeof(Token)) == cudaErrorMemoryAllocation) //Allocate new token list
			return Initialise_Device_List_MemAlloc;
		if (cudaMemcpy(globalList, list.formula(), list.count() * sizeof(Token), cudaMemcpyHostToDevice) != cudaSuccess) //Copy from host to device
			return Initialise_Copy_List_Error;
	}
	//Update new width and height
	globalWidth = width;
	globalHeight = height;
	globalListCount = list.count();
	globalHostResults = results;
	return Success;
}

__host__ ERRORCODES entryCalculate(Complex min, Complex max)
{
	if (log_str != nullptr)
		log_str("Entry Calculate\n");

	int N = globalWidth * globalHeight;
	ERRORCODES err = Success;

	auto first = Complex(min.real(), max.imag());

	auto last = Complex(max.real(), min.imag());

	if (!isHost)
	{
		if (log_str != nullptr)
			log_str("	Double-precision Kernel Execution\n");

		if (isShared)
			sharedCalculatef << < N / threadCount + (N % threadCount ? 1 : 0), threadCount, threadCount * globalMaxStack * sizeof(Complex) >> >(globalDeviceResults, globalList, globalListCount, globalMaxStack, first, last - first, globalWidth, globalHeight, globalWidth * globalHeight);
		else
			globalCalculate << < N / threadCount + (N % threadCount ? 1 : 0), threadCount >> >(globalDeviceResults, globalStack, globalList, globalListCount, globalMaxStack, first, last - first, globalWidth, globalHeight, globalWidth * globalHeight);
		
		if (cudaMemcpy(globalHostResults, globalDeviceResults, N*sizeof(RGB), cudaMemcpyDeviceToHost) != cudaSuccess)
			err = Calculate_Copy_Results_Error;
		else
			err = Success;

	}
	else
	{
		hostCalculate(globalHostResults, globalStack, globalList, globalListCount, globalMaxStack, first, last - first, globalWidth, globalHeight);
	}

	return err;
}

__host__ ERRORCODES entryDestruct()
{
	if (log_str != nullptr)
		log_str("Entry Destruct\n");

	ERRORCODES err = Success;
	if (isHost)
	{
		//delete[] globalStack;
	}
	else
	{
		if (cudaFree(globalList) == cudaErrorInvalidDevicePointer)
			err = Destruct_Device_List_InvalidPointer;
		if (cudaFree(globalDeviceResults) == cudaErrorInvalidDevicePointer)
			err = Destruct_Device_DeviceResults_InvalidPointer;
		if (cudaFree(globalStack) == cudaErrorInvalidDevicePointer)
			err = Destruct_Device_Stack_InvalidPointer;
	}
	return err;
}

__host__ void entryTranslate()
{

}

__host__ void entryTrace(Complex variable, TokenList list, Complex *ans, RGB *col = nullptr, double* mod = nullptr, double* arg = nullptr)
{
	auto stackmax = list.getStackSize();
	Complex* stack = new Complex[stackmax];

	--stack; //Stack is a before the beginning pointer, so decrement before use

	*ans = calculate(variable, list.formula(), list.count(), stack, 1);

	if (mod != nullptr)
		*mod = thrust::abs(*ans);
	if (arg != nullptr)
		*arg = thrust::arg(*ans);
	if (col != nullptr)
		*col = color(*ans);

	//delete stack;
}

/*__host__ Complex entryGradient(TokenList list, Complex value)
{
	return Complex();
}*/

__host__ Complex entryNewtonRaphson(TokenList list, Complex xn, int timeout)
{
	if (timeout-- == 0)
		return xn;

	auto next = xn - traceWrapper(list, xn) / fast_gradient(list, xn);

	if (thrust::abs(next - xn) < MAGIC)
		return next;

	return entryNewtonRaphson(list, next, timeout);
}
