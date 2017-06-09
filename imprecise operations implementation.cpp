#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define P_TH	8 << 23
#define N_TH	-(8 << 23)

float mul(float a, float b);
float sum(float a, float b);
float sub(float a, float b);
float log2(float x);
float _sqrt(float x);
float inverse(float x);
int __float_as_int(float f);
float __int_as_float(int i);


int main()
{
	float test1,test2;
	for(int i = 0; i < 10000; i++)
	{
		test1 = 100.0f / rand();
		test2 = 100.0f / rand();
		float mul_inacc = mul(test1, test2);
		float mul_acc = test1 * test2;
		float sum_inacc = sum(test1, test2);
		float sum_acc = test1 + test2;
		float sub_inacc = sub(test1, test2);
		float sub_acc = test1 - test2;
		test1 = fabs(test1);
		while(test1 > 1)	test1 *= 0.5;
		while(test1 < 0.5)	test1 *= 2;
		float log2_inacc = log2(fabs(test1));
		float log2_acc = log2f(fabs(test1));
		float sqrt_inacc = _sqrt(fabs(test1));
		float sqrt_acc = sqrtf(fabs(test1));
		float inv_inacc = inverse(test1);
		float inv_acc = 1 / test1;

		if(mul_inacc / mul_acc < 1 - 0.25f || mul_acc / mul_inacc < 1 - 0.25f)
			printf("mul : %f vs %f\n", mul_acc, mul_inacc);
		if(sum_inacc / sum_acc < 1 - 0.25f || sum_acc / sum_inacc < 1 - 0.25f)
			printf("sum : %f vs %f\n", sum_acc, sum_inacc);
		if(sub_inacc / sub_acc < 1 - 0.25f || sub_acc / sub_inacc < 1 - 0.25f)
			printf("sub : %f vs %f (%f, %f)\n", sub_acc, sub_inacc, test1, test2);
		if(log2_inacc / log2_acc < 1 - 0.20f || log2_acc / log2_inacc < 1 - 0.20f)
			printf("log2 : %f vs %f, (%f)\n", log2_acc, log2_inacc, 1 - log2_inacc / log2_acc);
		if(sqrt_inacc / sqrt_acc < 1 - 0.12f || sqrt_acc / sqrt_inacc < 1 - 0.12f)
			printf("sqrt : %f vs %f, (%f)\n", sqrt_acc, sqrt_inacc, 1 - sqrt_inacc / sqrt_acc);
		if(inv_inacc / inv_acc < 1 - 0.06f || inv_acc / inv_inacc < 1 - 0.06f)
			printf("inv : %f vs %f, (%f)\n", inv_acc, inv_inacc, 1 - inv_inacc / inv_acc);
	}

	return 0;
}

float mul(float a, float b)
{
	int m_sum	= (__float_as_int(a) & 0x007FFFFF) + (__float_as_int(b) & 0x007FFFFF);
	int m_new	= m_sum + 0x800000;

	if(m_sum & 0x800000) // M_a + M_b >= 1
	{
		m_new >>= 1;
		m_new += 0x800000;
	}
	
	m_new += (__float_as_int(a) & 0x7f800000) + (__float_as_int(b) & 0x7f800000) - 0x40000000; // exponent
	m_new |= (__float_as_int(a) & 0x80000000) ^ (__float_as_int(b) & 0x80000000); // sign

	return __int_as_float(m_new);
}


float sum(float a, float b)
{
	int d = (__float_as_int(a) & 0x7f800000) - (__float_as_int(b) & 0x7f800000);
	if(d > P_TH) return a; // abs(a) >>> abs(b)
	if(d < N_TH) return b; // abs(b) >>> abs(a)
	return a + b;
}


float sub(float a, float b)
{
	int d = (__float_as_int(a) & 0x7f800000) - (__float_as_int(b) & 0x7f800000);
	if(d > P_TH) return a; // abs(a) >>> abs(b)
	if(d < N_TH) return -b; // abs(b) >>> abs(a)
	return a - b;
}

float log2(float x)
{
	return ((__float_as_int(x) >> 23) - 0x80) + mul(0.9846f, x) - 0.9196f;
}

float _sqrt(float x)
{
	return x * (2.08f - 1.1911f * x);
}

float inverse(float x)
{
	return 2.823f - 1.882f * x;
}

union
{
	float f;
	int i;
} conv;

int __float_as_int(float f)
{
	conv.f = f;
	return conv.i;
}

float __int_as_float(int i)
{
	conv.i = i;
	return conv.f;
}