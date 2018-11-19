#ifndef PPG_SETUP_H
#define PPG_SETUP_H
#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#define NEAREST_MULTIPLE(n, d) (n - n % d)
typedef struct PPG_Params {
	size_t MAX_SAMPLES;			// maximum number of samples we can handle

	double f0;					// sample rate of original signal, in Hz
	double T0;					// 1 / f0, so in seconds

	double T_window;			// Window length for reconstruction algorithm,
								// in seconds
	size_t N_window;			// #samples in that window

	unsigned int CF;			// compression factor
	size_t M;					// compressed window length = N_window / CF

	double LASSO_lambda;		// regularization parameter for LASSO
	double CD_diff_thresh;		// stopping threshold for coordinate descent
								// algorithm
} PPG_Params;

PPG_Params get_ppg_params();
//void get_random_sample_flags_from_file(char *filepath, size_t N, bool *flags);
void ppg(double t0[10000], double X0[10000], double Xr[10000]);
//void dot_sub(double A[20*240], double s[240], double y_hat[20]);

#endif
