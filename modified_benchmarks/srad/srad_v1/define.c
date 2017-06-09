//====================================================================================================100
//====================================================================================================100
//	DEFINE
//====================================================================================================100
//====================================================================================================100

#define fp float

#define NUMBER_THREADS 512


//////////////////////////////////////////////////////////////////

#define E_MAX   0.30f
#define P_TH   8 << 23
#define N_TH   -(8 << 23)

#define AB_DIV(a,b) a*(2.823f-1.882f*b)
#define X_DIV(x) (2.823f-1.882f*x)


inline __device__ float mul(float a, float b)
{
   int m_sum   = (__float_as_int(a) & 0x007FFFFF) + (__float_as_int(b) & 0x007FFFFF);
   int m_new   = m_sum + 0x800000;

   if(m_sum & 0x800000) // M_a + M_b >= 1
   {
      m_new >>= 1;
      m_new += 0x800000;
   }
   
   m_new += (__float_as_int(a) & 0x7f800000) + (__float_as_int(b) & 0x7f800000) - 0x40000000; // exponent
   m_new |= (__float_as_int(a) & 0x80000000) ^ (__float_as_int(b) & 0x80000000); // sign

   return __int_as_float(m_new);
}


inline __device__ float sum(float a, float b)
{
   int d = (__float_as_int(a) & 0x7f800000) - (__float_as_int(b) & 0x7f800000);
   if(d > P_TH) return a; // abs(a) >>> abs(b)
   if(d < N_TH) return b; // abs(b) >>> abs(a)
   return a + b;
}


inline __device__ float sub(float a, float b)
{
   int d = (__float_as_int(a) & 0x7f800000) - (__float_as_int(b) & 0x7f800000);
   if(d > P_TH) return a; // abs(a) >>> abs(b)
   if(d < N_TH) return -b; // abs(b) >>> abs(a)
   return a - b;
}





//////////////////////////////////////////////////////////////////
