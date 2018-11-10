#ifndef PPG_SETUP_H
#define PPG_SETUP_H

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

PPG_Params get_ppg_params() {
	PPG_Params params;

	params.MAX_SAMPLES = 10000;
	params.f0 = 4.0;
	params.T0 = 1.0 / params.f0;

	params.T_window = 60;
	params.N_window = NEAREST_MULTIPLE(
							(size_t) floor(params.T_window / params.T0),
							4);
	params.CF = 12;
	params.M = params.N_window / params.CF;

	params.LASSO_lambda = 1.0e-2;
	params.CD_diff_thresh = 1.0e-2;

	return params;
}

#endif
