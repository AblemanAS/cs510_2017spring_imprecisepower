#include "srad.h"
#include <stdio.h>
#include <stdlib.h>

#define E_MAX   0.30f
#define P_TH   8 << 23
#define N_TH   -(8 << 23)

#define AB_DIV(a,b) a*(2.823f-1.882f*b)
#define X_DIV(x) (2.823f-1.882f*x)


//Additional Function
inline __device__ float mul(float a, float b)
{
	int m_sum = (__float_as_int(a) & 0x007FFFFF) + (__float_as_int(b) & 0x007FFFFF);
	int m_new = m_sum + 0x800000;

	if (m_sum & 0x800000) // M_a + M_b >= 1
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
	if (d > P_TH) return a; // abs(a) >>> abs(b)
	if (d < N_TH) return b; // abs(b) >>> abs(a)
	return a + b;
}


inline __device__ float sub(float a, float b)
{
	int d = (__float_as_int(a) & 0x7f800000) - (__float_as_int(b) & 0x7f800000);
	if (d > P_TH) return a; // abs(a) >>> abs(b)
	if (d < N_TH) return -b; // abs(b) >>> abs(a)
	return a - b;
}
//

__global__ void
srad_cuda_1(
	float *E_C,
	float *W_C,
	float *N_C,
	float *S_C,
	float * J_cuda,
	float * C_cuda,
	int cols,
	int rows,
	float q0sqr
)
{

	//block id
	int bx = blockIdx.x;
	int by = blockIdx.y;

	//thread id
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//indices
	int index = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
	int index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
	int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
	int index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
	int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

	float n, w, e, s, jc, g2, l, num, den, qsqr, c;

	//shared memory allocation
	__shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE];

	__shared__ float north[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float south[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float  east[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float  west[BLOCK_SIZE][BLOCK_SIZE];

	//load data to shared memory
	north[ty][tx] = J_cuda[index_n];
	south[ty][tx] = J_cuda[index_s];
	if (by == 0) {
		north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx];
	}
	else if (by == gridDim.y - 1) {
		south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
	}
	__syncthreads();

	west[ty][tx] = J_cuda[index_w];
	east[ty][tx] = J_cuda[index_e];

	if (bx == 0) {
		west[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
	}
	else if (bx == gridDim.x - 1) {
		east[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) + cols * ty + BLOCK_SIZE - 1];
	}

	__syncthreads();



	temp[ty][tx] = J_cuda[index];

	__syncthreads();

	jc = temp[ty][tx];

	if (ty == 0 && tx == 0) { //nw
		n = sub(north[ty][tx], jc);
		s = sub(temp[ty + 1][tx], jc);
		w = sub(west[ty][tx], jc);
		e = sub(temp[ty][tx + 1], jc);
	}
	else if (ty == 0 && tx == BLOCK_SIZE - 1) { //ne
		n = sub(north[ty][tx], jc);
		s = sub(temp[ty + 1][tx], jc);
		w = sub(temp[ty][tx - 1], jc);
		e = sub(east[ty][tx], jc);
	}
	else if (ty == BLOCK_SIZE - 1 && tx == BLOCK_SIZE - 1) { //se
		n = sub(temp[ty - 1][tx], jc);
		s = sub(south[ty][tx], jc);
		w = sub(temp[ty][tx - 1], jc);
		e = sub(east[ty][tx], jc);
	}
	else if (ty == BLOCK_SIZE - 1 && tx == 0) {//sw
		n = sub(temp[ty - 1][tx], jc);
		s = sub(south[ty][tx], jc);
		w = sub(west[ty][tx], jc);
		e = sub(temp[ty][tx + 1], jc);
	}

	else if (ty == 0) { //n
		n = sub(north[ty][tx], jc);
		s = sub(temp[ty + 1][tx], jc);
		w = sub(temp[ty][tx - 1], jc);
		e = sub(temp[ty][tx + 1], jc);
	}
	else if (tx == BLOCK_SIZE - 1) { //e
		n = sub(temp[ty - 1][tx], jc);
		s = sub(temp[ty + 1][tx], jc);
		w = sub(temp[ty][tx - 1], jc);
		e = sub(east[ty][tx], jc);
	}
	else if (ty == BLOCK_SIZE - 1) { //s
		n = sub(temp[ty - 1][tx], jc);
		s = sub(south[ty][tx], jc);
		w = sub(temp[ty][tx - 1], jc);
		e = sub(temp[ty][tx + 1], jc);
	}
	else if (tx == 0) { //w
		n = sub(temp[ty - 1][tx], jc);
		s = sub(temp[ty + 1][tx], jc);
		w = sub(west[ty][tx], jc);
		e = sub(temp[ty][tx + 1], jc);
	}
	else {  //the data elements which are not on the borders 
		n = sub(temp[ty - 1][tx], jc);
		s = sub(temp[ty + 1][tx], jc);
		w = sub(temp[ty][tx - 1], jc);
		e = sub(temp[ty][tx + 1], jc);
	}


	//g2 = (n * n + s * s + w * w + e * e) / (jc * jc);
	g2 = AB_DIV(sum(sum(mul(n, n), mul(s, s)), sum(mul(w, w), mul(e, e))), mul(jc, jc));
	//l = (n + s + w + e) / jc;
	l = AB_DIV(sum(sum(n, s), sum(w, e)), jc);

	num = (0.5*g2) - ((1.0 / 16.0)*(l*l));
	den = 1 + (.25*l);
	qsqr = AB_DIV(num, mul(den,den));

	// diffusion coefficent (equ 33)
	//den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
	den = AB_DIV(sub(qsqr, q0sqr), mul(q0sqr, 1 + q0sqr));

	//c = 1.0 / (1.0 + den);
	c = X_DIV(1.0 + den);
	
	// saturate diffusion coefficent
	if (c < 0) { temp_result[ty][tx] = 0; }
	else if (c > 1) { temp_result[ty][tx] = 1; }
	else { temp_result[ty][tx] = c; }

	__syncthreads();

	C_cuda[index] = temp_result[ty][tx];
	E_C[index] = e;
	W_C[index] = w;
	S_C[index] = s;
	N_C[index] = n;

}

__global__ void
srad_cuda_2(
	float *E_C,
	float *W_C,
	float *N_C,
	float *S_C,
	float * J_cuda,
	float * C_cuda,
	int cols,
	int rows,
	float lambda,
	float q0sqr
)
{
	//block id
	int bx = blockIdx.x;
	int by = blockIdx.y;

	//thread id
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//indices
	int index = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
	int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
	int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
	float cc, cn, cs, ce, cw, d_sum;

	//shared memory allocation
	__shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float  east_c[BLOCK_SIZE][BLOCK_SIZE];

	__shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];

	//load data to shared memory
	temp[ty][tx] = J_cuda[index];

	__syncthreads();

	south_c[ty][tx] = C_cuda[index_s];

	if (by == gridDim.y - 1) {
		south_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
	}
	__syncthreads();


	east_c[ty][tx] = C_cuda[index_e];

	if (bx == gridDim.x - 1) {
		east_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * (gridDim.x - 1) + cols * ty + BLOCK_SIZE - 1];
	}

	__syncthreads();

	c_cuda_temp[ty][tx] = C_cuda[index];

	__syncthreads();

	cc = c_cuda_temp[ty][tx];

	if (ty == BLOCK_SIZE - 1 && tx == BLOCK_SIZE - 1) { //se
		cn = cc;
		cs = south_c[ty][tx];
		cw = cc;
		ce = east_c[ty][tx];
	}
	else if (tx == BLOCK_SIZE - 1) { //e
		cn = cc;
		cs = c_cuda_temp[ty + 1][tx];
		cw = cc;
		ce = east_c[ty][tx];
	}
	else if (ty == BLOCK_SIZE - 1) { //s
		cn = cc;
		cs = south_c[ty][tx];
		cw = cc;
		ce = c_cuda_temp[ty][tx + 1];
	}
	else { //the data elements which are not on the borders 
		cn = cc;
		cs = c_cuda_temp[ty + 1][tx];
		cw = cc;
		ce = c_cuda_temp[ty][tx + 1];
	}

	// divergence (equ 58)
	//d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];
	d_sum = sum(sum(mul(cn, N_C[index]), mul(cs, S_C[index])), sum(mul(cw, W_C[index]), mul(ce, E_C[index])));

	// image update (equ 61)
	//c_cuda_result[ty][tx] = temp[ty][tx] + 0.25 * lambda * d_sum;
	c_cuda_result[ty][tx] = sum(temp[ty][tx], 0.25 * mul(lambda, d_sum));
	__syncthreads();

	J_cuda[index] = c_cuda_result[ty][tx];

}
