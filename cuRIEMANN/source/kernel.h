#ifndef KERNEL_H
#define KERNEL_H


#include <thrust\complex.h>


#ifdef COODA_EXPORTS
#define COODA_EXPORTS __declspec(dllexport) 
#else
#define COODA_EXPORTS __declspec(dllimport) 
#endif

#ifdef __cplusplus
extern "C" {
#endif

	typedef void(*LOG_STRING)(char*);

	typedef double  SinglePrecision;
	typedef double DoublePrecision;

	typedef thrust::complex<double> Complex; 

	struct RGB { unsigned char a;  unsigned char r; unsigned char g; unsigned char b; };

	struct Token { int type; Complex data; };

	const double MAGIC = 2.16e-13; //Modified machine-epsilon value, changed to work with thrust trig functions.

	class TokenList
	{
	private:
		Token* m_pFormula;
		unsigned m_iCount;

	public:
		TokenList() : m_pFormula(nullptr), m_iCount(0) {}
		TokenList(unsigned s, Token* f) : m_pFormula(f), m_iCount(s) {}

		unsigned count() const { return m_iCount; }
		unsigned type(unsigned i) const { return m_pFormula[i].type; }
		const Complex& data(unsigned i) const { return m_pFormula[i].data; }

		Token* formula() const { return m_pFormula; }

		unsigned getStackSize() const
		{
			unsigned maxSize = 1;
			unsigned stackSize = 0;

			for (unsigned i = 0; i < m_iCount; ++i)
			{
				if (m_pFormula[i].type == 2 && m_pFormula[i].data.real() < 6) //If item is a binary operator, 2 items will be popped one pushed, with a net movement of -1
					stackSize--;
				else if (m_pFormula[i].type == 1 || m_pFormula[i].type == 3) //If the item is z or a constant, 1 item will be pushed, with a net movement of +1
					stackSize++;
				//Otherise the item is a unary operator, which pops one item and pushes one item, not affecting the size.
				maxSize = stackSize > maxSize ? stackSize : maxSize;
			}
			return maxSize;
		}
	};

	enum ERRORCODES
	{
		Success = 0,
		Construct_InvalidDevice,
		Construct_Test_MemAlloc,
		Construct_Test_InvalidPointer,

		Initialise_Host_HostResults_MemAlloc,
		Initialise_Host_Stack_MemAlloc,

		Initialise_Device_DeviceResults_InvalidPointer,
		Initialise_Device_DeviceResults_MemAlloc,
		Initialise_Device_Stack_InvalidPointer,
		Initialise_Device_Stack_MemAlloc,
		Initialise_Device_List_InvalidPointer,
		Initialise_Device_List_MemAlloc,

		Initialise_Copy_List_Error,

		Calculate_Copy_Results_Error,

		Destruct_Device_List_InvalidPointer,
		Destruct_Device_DeviceResults_InvalidPointer,
		Destruct_Device_Stack_InvalidPointer

	};

	extern "C" __declspec(dllexport) ERRORCODES entryConstruct(bool, LOG_STRING);

	extern "C" __declspec(dllexport) ERRORCODES entryInitialise(unsigned width, unsigned height, TokenList list, RGB* results);
	
	extern "C" __declspec(dllexport) ERRORCODES entryCalculate(Complex min, Complex max);

	extern "C" __declspec(dllexport) ERRORCODES entryDestruct();

	extern "C" __declspec(dllexport) void entryTranslate();
		
	extern "C" __declspec(dllexport) void entryTrace(Complex, TokenList, Complex*, RGB*, double*, double*);
	
	//extern "C" __declspec(dllexport) Complex entryGradient(TokenList, Complex);

	extern "C" __declspec(dllexport) Complex entryNewtonRaphson(TokenList, Complex, int);

#ifdef __cplusplus
}
#endif

#endif 

//Complex* dllEntry(Token* Tokens, int tokenLength, int canvasWidth, int canvasHeight, Complex domainMin, Complex domainMax)