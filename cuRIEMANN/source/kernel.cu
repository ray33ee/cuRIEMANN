#include <stdio.h>

#include <math.h>
#include <new> //std::nothrow

#include "Complex.h"

#include "kernel.h"

int const threadCount = 192; //Number of threads per block

//Dimensions
unsigned globalWidth;
unsigned globalHeight;

//Flags
bool isHost; //host = true, false = device
bool isShared; //shared memory = true, global memory = false

bool isVerbose; //For debugging purposes

//Arrays
Token* globalList;
RGB* globalDeviceResults;
RGB* globalHostResults;
thrust::complex<SinglePrecision>* globalStack;
thrust::complex<DoublePrecision>* globalDoubleStack;

unsigned globalListCount;
unsigned globalMaxStack;

//Array sizes
unsigned resultsSize;
unsigned stackSize;

__host__ ERRORCODES entryConstruct(int flags = 0)
{
	int deviceCount;

	isVerbose = (flags & 0x1);

	printf("Flag list:\n");
	printf("Verbose: ");
	if (flags & 0x1)
		printf("On\n");
	else
		printf("Off\n");

	printf("Force Host: ");
	if (flags & 0x2)
		printf("On\n");
	else
		printf("Off\n");

	if (isVerbose)
		printf("Entry Construct\n");

	cudaError_t firstCall = cudaGetDeviceCount(&deviceCount); 

	if (flags & 0x2) //Force host
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
			if (isVerbose)
				printf("	Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
		}

		{
			int* a;
			if (cudaMalloc(&a, sizeof(int)) == cudaErrorMemoryAllocation)
				return Construct_Test_MemAlloc;
			if (cudaFree(a) == cudaErrorInvalidDevicePointer)
				return Construct_Test_InvalidPointer;
		}
		if (isVerbose)
			printf("	Using device...\n");

	}
	else //Otherwise use host
	{
		if (isVerbose)
			printf("	Using Host...\n");

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

__host__ ERRORCODES entryInitialise(unsigned width, unsigned height, TokenList* list, RGB* results)
{
	globalMaxStack = list->getStackSize();

	if (isVerbose)
		printf("Entry Initialise\n");

	if (isHost) //Uses - results, d_stack, globalList
	{
		if (stackSize < width * height * sizeof(thrust::complex<SinglePrecision>) * globalMaxStack) //Current stack array is too small, reallocate
		{
			if (isVerbose)
				printf("	Reallocate host stack - Host\n");
			//delete[] globalStack; //Delete current stack array if allocated
			globalStack = new (std::nothrow) thrust::complex<SinglePrecision>[width*height*globalMaxStack]; //Construct new stack array
			if (globalStack == nullptr)
				return Initialise_Host_Stack_MemAlloc;
			stackSize = width * height * globalMaxStack * sizeof(thrust::complex<SinglePrecision>); //Update stackSize
		}
		//delete[] globalList; //free current token list
		globalList = list->formula(); //Update new tokenlist
	}
	else //Uses - globalList, d_results, results and d_stack
	{
		if (resultsSize < width * height * sizeof(thrust::complex<SinglePrecision>)) //Current results array is too small, reallocate
		{
			if (isVerbose)
			{
				printf("	Reallocate device results - Device - old size: %i, new size: %i\n", resultsSize, width * height * sizeof(thrust::complex<SinglePrecision>));
				printf("	width: %i, height: %i, sizeof: %i\n", width, height, sizeof(thrust::complex<SinglePrecision>));
			}
			if (cudaFree(globalDeviceResults) == cudaErrorInvalidDevicePointer) //Delete current device results array if allocated
				return Initialise_Device_DeviceResults_InvalidPointer;
			if (cudaMalloc(&globalDeviceResults, width * height * sizeof(RGB)) == cudaErrorMemoryAllocation) //Construct new device array
				return Initialise_Device_DeviceResults_MemAlloc;
			resultsSize = width * height * sizeof(RGB); //Update resultSize
		}

		if (isVerbose)
			printf("	Stack max: %i\n", globalMaxStack);
		isShared = globalMaxStack < 5;

		if (!isShared && stackSize < width * height * sizeof(thrust::complex<SinglePrecision>) * globalMaxStack) //If we're using global memory and the memory needs expanding...
		{
			if (isVerbose)
				printf("	Reallocate device stack - device\n");
			if (cudaFree(globalStack) == cudaErrorInvalidDevicePointer) //Delete current stack array if allocated
				return Initialise_Device_Stack_InvalidPointer;
			if (cudaMalloc(&globalStack, width * height * globalMaxStack * sizeof(thrust::complex<SinglePrecision>)) == cudaErrorMemoryAllocation) //Construct new stack array
				return Initialise_Device_Stack_MemAlloc;
			stackSize = width * height * globalMaxStack * sizeof(thrust::complex<SinglePrecision>); //Update stackSize
		}
		if (isVerbose)
			printf("	Copying list to device\n");
		if (cudaFree(globalList) == cudaErrorInvalidDevicePointer) //Free device token list
			return Initialise_Device_List_InvalidPointer;
		if (cudaMalloc(&globalList, list->count() * sizeof(Token)) == cudaErrorMemoryAllocation) //Allocate new token list
			return Initialise_Device_List_MemAlloc;
		if (cudaMemcpy(globalList, list->formula(), list->count() * sizeof(Token), cudaMemcpyHostToDevice) != cudaSuccess) //Copy from host to device
			return Initialise_Copy_List_Error;
	}
	//Update new width and height
	globalWidth = width;
	globalHeight = height;
	globalListCount = list->count();
	globalHostResults = results;
	return Success;
}

__host__ ERRORCODES entryCalculate(thrust::complex<SinglePrecision> min, thrust::complex<SinglePrecision> max)
{
	if (isVerbose)
		printf("Entry Calculate\n");

	int N = globalWidth * globalHeight;
	ERRORCODES err = Success;

	if (!isHost)
	{
		if (isVerbose)
			printf("	Single-precision Kernel Execuation\n");

		if (isShared)
			sharedCalculatef << < N / threadCount + (N % threadCount ? 1 : 0), threadCount, threadCount * globalMaxStack * sizeof(thrust::complex<SinglePrecision>) >> >(globalDeviceResults, globalList, globalListCount, globalMaxStack, min, max - min, globalWidth, globalHeight, globalWidth * globalHeight);
		else
			globalCalculate << < N / threadCount + (N % threadCount ? 1 : 0), threadCount >> >(globalDeviceResults, globalStack, globalList, globalListCount, globalMaxStack, min,max - min, globalWidth, globalHeight, globalWidth * globalHeight);
		
		if (cudaMemcpy(globalHostResults, globalDeviceResults, N*sizeof(RGB), cudaMemcpyDeviceToHost) != cudaSuccess)
			err = Calculate_Copy_Results_Error;
		else
			err = Success;

	}
	else
	{
		hostCalculate(globalHostResults, globalStack, globalList, globalListCount, globalMaxStack, min, max - min, globalWidth, globalHeight);
	}

	return err;
}

__host__ ERRORCODES entryDestruct()
{
	if (isVerbose)
		printf("Entry Destruct\n");

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

__host__ void entryTrace(thrust::complex<DoublePrecision> variable, RGB &col, thrust::complex<DoublePrecision> &ans, TokenList* list, double& mod, double& arg)
{
	auto stackmax = list->getStackSize();
	thrust::complex<DoublePrecision>* stack = new thrust::complex<DoublePrecision>[stackmax];

	--stack; //Stack is a before the beginning pointer, so decrement before use

	ans = calculate(variable, list->formula(), list->count(), stack, 1);
	mod = thrust::abs(ans);
	arg = thrust::arg(ans);
	col = color(ans);

	//delete stack;
}

int main()
{
	ERRORCODES error;

	entryConstruct(true);

	auto answer = new RGB[1920*1080];

	TokenList* list;
	unsigned count = 3;

	Token* tokens = new Token[count];

	tokens[0] = { 1, { 0, 0 } };
	tokens[1] = { 1, { 0, 0 } };
	tokens[2] = { 2, { 4, 0 } };

	list = new TokenList(count, tokens);

	entryInitialise(1920, 1080, list, answer); //1920 x 1080 f(z) = z Quickest, simplest full HD graph

	error = entryCalculate({ -200.0f, -200.0f }, { 300.0f, 300.0f });

	RGB color; 
	thrust::complex<double> z;
	double mod, arg;

	entryTrace({ 3.0, 0.0 }, color, z, list, mod, arg);

	printf("answer: %f, %f - %f, %f\n", z.real(), z.imag(), mod, arg);

	//entryInitialise(1920, 1080, TokenList<Precision>({ { 1, { 0, 0 } }, { 1, { 1, 0 } }, { 2, { 4, 0 } } })); //1920 x 1080 f(z) = z ^ z Binary operation full HD graph

	//answer = entryCalculate({ -2.0f, -2.0f }, { 2.0f, 2.0f }, error);
	
	//entryInitialise(1920, 1080, TokenList<Precision>({ { 1, { 0, 0 } }, { 1, { 0, 0 } }, { 1, { 0, 0 } }, { 1, { 0, 0 } }, { 1, { 0, 0 } }, { 1, { 0, 0 } }, { 2, { 1, 0 } }, { 2, { 1, 0 } }, { 2, { 1, 0 } }, { 2, { 1, 0 } }, { 2, { 1, 0 } } })); //1920 x 1080 f(z) = z + z + z + z + Z Tests globalCalculate kernel

	//entryCalculate({ -2.0f, -2.0f }, { 2.0f, 2.0f }, error);
	
	//entryInitialise(1920 * 2, 1080 * 2, TokenList<Precision>({ { 3, { 1, 0 } }, { 1, { 0, 0 } }, { 1, { 0, 0 } }, { 2, { 2, 0 } }, { 2, { 1, 0 } }, { 2, { 9, 0 } } })); //4k f(z) = ln(z^2 + 1) //Somewhat difficult 4K graph

	//answer = entryCalculate({ -2.0f, -2.0f }, { 2.0f, 2.0f }, error);

	//entryInitialise(1920 * 2, 1080 * 2, TokenList<Precision>({ { 1, { 0, 0 } }, { 3, { 1, 0 } }, { 2, { 4, 0 } }, { 2, { 23, 0 } }, { 1, { 0, 0 } }, { 3, { 2, 0 } }, { 2, { 4, 0 } }, { 2, { 23, 0 } }, { 2, { 1, 0 } } })); //4k First few terms of riemann zeta function //Somewhat difficult 4K graph
	
	//answer = entryCalculate({ -2.0f, -2.0f }, { 2.0f, 2.0f }, error);

	//entryInitialise(1920, 1080, TokenList<Precision>({ { 1, { 0, 0 } }, { 1, { 0, 0 } }, { 2, { 4, 0 } } })); //1920 x 1080 f(z) = ln(z) Unary operation full HD graph

	//answer = entryCalculate({ 2.0f, 2.0f }, { 1002.0f, 1002.0f }, error);

	for (int i = 0; i < 10; i++)
		printf("%i %i %i %i - ind: %i\n", (int)answer[i].a, (int)answer[i].r, (int)answer[i].g, (int)answer[i].b, i);


	//for (int i = 1920 * 1080 - 10; i < 1920 * 1080; i++)
	//	printf("(%i) %i %i %i - ind: %i\n", (int)answer[i].a, (int)answer[i].r, (int)answer[i].g, (int)answer[i].b, i);

	cudaDeviceReset();

	while (1);
}