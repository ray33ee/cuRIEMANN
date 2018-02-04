#include <math.h>

#include <initializer_list>

#include "kernel.h"

#include <thread>
#include <vector>
#include <algorithm>

template <class T> 
__device__ __host__ static RGB HLtoRGB(T &h, T &l)
{
	typedef unsigned char byte;

	T v;

	//As the argument could be pi radians, the hue could be 1.0. As this is an invalid value, this can be fixed by making the hue 0, which (as the hue is cyclic) is the same
	if (h == 1.0f)
		h = 0;

	v = (l <= 0.5f) ? l * 2 : 1;

	if (v > 0)
	{
		T m;
		T sv;
		int sextant;
		T vsf, mid1, mid2;

		m = l + l - v;
		sv = (v - m) / v;
		h *= 6.0f;
		sextant = (int)h;
		vsf = v * sv * (h - sextant);
		mid1 = m + vsf;
		mid2 = v - vsf;
		switch (sextant)
		{
		case 5:
			return{ (byte)(v * 255), (byte)(mid1 * 255), (byte)(m * 255), 255 };
		case 0:
			return{ (byte)(mid2 * 255), (byte)(v * 255), (byte)(m * 255), 255 };
		case 1:
			return{ (byte)(m * 255), (byte)(v * 255), (byte)(mid1 * 255), 255 };
		case 2:
			return{ (byte)(m * 255), (byte)(mid2 * 255), (byte)(v * 255), 255 };
		case 3:
			return{ (byte)(mid1 * 255), (byte)(m * 255), (byte)(v * 255), 255 };
		case 4:
			return{ (byte)(v * 255), (byte)(m * 255), (byte)(mid2 * 255), 255 };
		}
	}
	//If the application gets here, there is a problem. Output the otherwise impossible grey to indicate error
	return{ 128, 128, 128, 255 };
}

template <class T>
__device__ __host__ RGB color(const thrust::complex<T> &z)
{
	//Magic numbers galore...
	T hue, lightness, modarg;

	hue = (thrust::arg(z) + 3.14159265359f) / 6.28318530718f;
	modarg = log(thrust::abs(z));

	

	if (modarg < 0)
	{
		lightness = 0.75f - thrust::abs(z) / 2.0f;
	}
	else
	{
		if (!((int)modarg & 1)) //If whole part of modarg is even, 0 --> 1 maps to black --> white
			lightness = (modarg - floor(modarg)) / 2.0f + 0.25f;
		else //If whole part of modarg is odd 0 --> 1 maps to white --> black
			lightness = 0.75f - (modarg - floor(modarg)) / 2.0f;
	}
	return HLtoRGB(hue, lightness);
}

template <class T>
__device__ __host__ const thrust::complex<T>& calculate(thrust::complex<T> z, Token* list, unsigned tokenCount, thrust::complex<T>* stackTop, unsigned stride)
{
	for (unsigned i = 0; i < tokenCount; ++i)
	{
		if (list[i].type == 1)
		{
			stackTop += stride;
			*stackTop = z;
		}
		else if (list[i].type == 2)
		{
			switch ((unsigned)list[i].data.real())
			{
			case 0:
				stackTop -= stride;
				*stackTop = *stackTop + *(stackTop + 1); //Equivalent to popping two operands from stack, subtracting them, and pushing the result.
				break;
			case 1:
				stackTop -= stride;
				*stackTop = *stackTop - *(stackTop + 1);
				break;
			case 2:
				stackTop -= stride;
				*stackTop = *stackTop * *(stackTop + 1);
				break;
			case 3:
				stackTop -= stride;
				*stackTop = *stackTop / *(stackTop + 1);
				break;
			case 4:
				stackTop -= stride;
				*stackTop = thrust::pow(*stackTop, *(stackTop + 1));
				break;
			case 5:
				stackTop -= stride;
				*stackTop = thrust::log(*stackTop) / thrust::log(*(stackTop + 1));
				break;
			case 6:
				*stackTop = -(*stackTop); //Equivalent to popping one operand, negating it, and pushing result to stack
				break;
			case 7:
				*stackTop = (thrust::conj(*stackTop));
				break;
			case 8:
				*stackTop = (thrust::sqrt(*stackTop));
				break;
			case 9:
				*stackTop = (thrust::log(*stackTop));
				break;
			case 10:
				*stackTop = (thrust::exp(*stackTop));
				break;
			case 11:
				*stackTop = (thrust::sinh(*stackTop));
				break;
			case 12:
				*stackTop = (thrust::cosh(*stackTop));
				break;
			case 13:
				*stackTop = (thrust::tanh(*stackTop));
				break;
			case 14:
				*stackTop = (thrust::sin(*stackTop));
				break;
			case 15:
				*stackTop = (thrust::cos(*stackTop));
				break;
			case 16:
				*stackTop = (thrust::tan(*stackTop));
				break;
			case 17:
				*stackTop = (thrust::asinh(*stackTop));
				break;
			case 18:
				*stackTop = (thrust::acosh(*stackTop));
				break;
			case 19:
				*stackTop = (thrust::atanh(*stackTop));
				break;
			case 20:
				*stackTop = (thrust::asin(*stackTop));
				break;
			case 21:
				*stackTop = (thrust::acos(*stackTop));
				break;
			case 22:
				*stackTop = (thrust::atan(*stackTop));
				break;
			case 23:
				*stackTop = thrust::complex<T>(1) / (*stackTop);
				break;
			case 24:
				*stackTop = thrust::abs(*stackTop);
				break;
			case 25:
				*stackTop = thrust::arg(*stackTop);
				break;
			}
		}
		else if (list[i].type == 3)
		{
			stackTop += stride;
			*stackTop = (list[i].data);
		}
	}
	return *stackTop;
}

template <class T>
__device__ __host__ void calculate(int ind, thrust::complex<T>* stackTop, RGB* results, Token* list, unsigned tokenCount, thrust::complex<T> firstDomain, thrust::complex<T> diffDomain, unsigned width, unsigned height, unsigned stackStride)
{
	
		thrust::complex<T> z = {
			firstDomain.real() + diffDomain.real() * (ind % width) / width,
			firstDomain.imag() + diffDomain.imag() * (ind / width) / height
		};

		thrust::complex<T> ans = calculate(z, list, tokenCount, stackTop, stackStride);

		
		results[ind] = color(ans);
		
	
}

// Function called concurrently by host threads
template <class T>
__host__ void concurrentCalculate(unsigned blockIndex, unsigned blockSize, RGB* results, thrust::complex<T>* stack, Token* list, unsigned tokenCount, unsigned stackMaxCount, thrust::complex<T> first, thrust::complex<T> diff, unsigned width, unsigned height)
{
	for (unsigned i = 0; i < blockSize; ++i)
	{
		auto index = blockIndex * blockSize + i;
		calculate(index, stack + index * stackMaxCount, results, list, tokenCount, first, diff, width, height, 1);
	}
}

template <class T>
__host__ void hostCalculate(RGB* results, thrust::complex<T>* stack, Token* list, unsigned tokenCount, unsigned stackMaxCount, thrust::complex<T> first, thrust::complex<T> diff, unsigned width, unsigned height)
{
	unsigned count = width * height;

	std::vector<std::thread> threads;

	auto hardwareThreadCount = std::thread::hardware_concurrency();

	auto threadsAvailable = hardwareThreadCount > 1 ? hardwareThreadCount - 1 : 1;

	auto blockSize = count / threadsAvailable;
	
	for (unsigned i = 0; i < threadsAvailable; ++i)
			threads.push_back(std::thread(concurrentCalculate<T>, i, blockSize + (count % threadsAvailable), results, stack, list, tokenCount, stackMaxCount, first, diff, width, height)); //Conditional statement used to assign final thread, which may contain a larger block if threadsToUse doesn't divide into height.

	for (auto &iter : threads)
		iter.join();
}

//Following kernel uses shared memory to contain stack
__global__ void sharedCalculatef(RGB* results, Token* list, unsigned tokenCount, unsigned stackMaxCount, Complex firstDomain, Complex diffDomain, unsigned width, unsigned height, unsigned N) //
{
	extern __shared__ Complex stackBlockf[];

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		auto stackTop = stackBlockf + threadIdx.x * stackMaxCount - 1; //Pointer to top element in stack
		calculate<SinglePrecision>(i, stackTop, results, list, tokenCount, firstDomain, diffDomain, width, height, 1);
	}
}

//Following kernel uses global memory to contain stack
template <class T>
__global__ void globalCalculate(RGB* results, thrust::complex<T>* stack, Token* list, unsigned tokenCount, unsigned stackMaxCount, thrust::complex<T> firstDomain, thrust::complex<T> diffDomain, unsigned width, unsigned height, unsigned N) //
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		auto stackTop = stack + i - width * height; //Pointer to top element in stack
		calculate<T>(i, stackTop, results, list, tokenCount, firstDomain, diffDomain, width, height, width * height);
	}
}

template <class T>
__host__ T traceWrapper(TokenList list, T value)
{
	T ans;
	entryTrace(value, list, &ans);
	return ans;
}

template <class T>
__host__ T fast_gradient(TokenList list, T value)
{
	//determine a suitably small value of h as sqrt(epsilon)*x. Since x=0 will result in divide by zero, replace x with suitably small value, i.e. x=epsilon.
	auto h = sqrt(MAGIC) * value;

	return (traceWrapper(list, value + h) - traceWrapper(list, value)) / h;
}