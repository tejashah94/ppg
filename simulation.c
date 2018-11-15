/**
 * Algorithm for PPG signal reconstruction from compressively sampled input.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#include "setup.h"

double max(double a, double b) {
	return (a > b ? a : b) ;
}

// TODO: implement fast DCT so this isn't needed.
void dct_1D(double *X, size_t N, double *X_DCT) {
	memset(X_DCT, 0, sizeof(double) * N);
	const double factor_1 = 2.0 / sqrt(N);
	const double factor_0 = factor_1 / sqrt(2);
	for (size_t k = 0; k < N; ++k) {
		for (size_t n = 0; n < N; ++n) {
			X_DCT[k] += cos((M_PI / N) * (n + 0.5) * k) * X[n];
		}
	}
	X_DCT[0] *= factor_0;
	for (size_t k = 1; k < N; ++k) {
		X_DCT[k] *= factor_1;
	}
}

void idct_1D(double *X_DCT, size_t N, double *X) {
	memset(X, 0, sizeof(double) * N);
	const double factor_1 = 2.0 / sqrt(N);
	const double factor_0 = factor_1 / sqrt(2);
	for (size_t n = 0; n < N; ++n) {
		X[n] += X_DCT[0] * factor_0;
	}

	for (size_t k = 1; k < N; ++k) {
		for (size_t n = 0; n < N; ++n) {
			X[n] += cos((M_PI / N) * (n + 0.5) * k) * X_DCT[k] * factor_1;
		}
	}
}

void random_sample_idct_1D(double *X_DCT, size_t N, bool *phi_flags, size_t M, double *Y) {
	memset(Y, 0, sizeof(double) * M);
	const double factor_1 = 2.0 / sqrt(N);
	const double factor_0 = factor_1 / sqrt(2);
	for (size_t m = 0; m < M; ++m) {
		Y[m] = X_DCT[0] * factor_0;
	}

	size_t m = 0;
	for (size_t n = 0; n < N; ++n) {
		if (phi_flags[n]) {
			for (size_t k = 1; k < N; ++k) {
				Y[m] += cos((M_PI / N) * (n + 0.5) * k) * X_DCT[k] * factor_1;
			}
			m++;
		}
	}
}

void vec_sub_update(double *vec, double *b, size_t N) {
	for (size_t i = 0; i < N; ++i) {
		vec[i] -= b[i];
	}
}

void calc_Anorm2_randsampleDCT(bool *phi_flags, size_t N, double *A_norm2) {
	memset(A_norm2, 0, sizeof(double) * N);
	const double factor_1 = 2.0 / sqrt(N);
	const double factor_0 = factor_1 / sqrt(2);
	for (size_t n = 0; n < N; ++n) {
		if (phi_flags[n]) {
			A_norm2[0] += factor_0 * factor_0;
			for (size_t k = 1; k < N; ++k) {
				A_norm2[k] += pow(cos((M_PI / N) * (n + 0.5) * k) * factor_1, 2);
			}
		}
	}
}

void cd_lasso_randsampleDCT(double *y, bool *phi_flags,
							size_t M, size_t N, double lambda,
							double *s_hat, double tol) {
	// Y is M-dim, phi_flags controls which DCT components to use, x_hat is Nx1
	// We don't need an explicit (A) matrix, as we know it's going to randomly
	// sample from the IDCT. We can call the functions which can calculate the
	// products we want from phi_flags and the definition of the DCT.
	double *A_norm2 = malloc(sizeof(double) * N);
	calc_Anorm2_randsampleDCT(phi_flags, N, A_norm2);
	
	// can initialize with anything I think...
	for (size_t k = 0; k < N; ++k) {
		s_hat[k] = 0.5;
	}

	double *y_hat = malloc(sizeof(double) * M);
	random_sample_idct_1D(s_hat, N, phi_flags, M, y_hat);
	for (size_t i = 0; i < M; ++i) {
		printf("%lf ", y_hat[i]);
	}
	printf("\n");

	double *r = malloc(sizeof(double) * M);
	memcpy(r, y, sizeof(double) * M);
	vec_sub_update(r, y_hat, M);

	const double factor_1 = 2.0 / sqrt(N);
	const double factor_0 = factor_1 / sqrt(2);
	
	double max_sk = 0.5; // depends on initialization
	while (true) {
		double max_dsk = 0;
		for (size_t k = 0; k < N; ++k) {
			if (A_norm2[k] == 0.0)
				continue;

			double factor = (k == 0 ? factor_0 : factor_1);
			double s_k0 = s_hat[k];

			// r += A[:,k] .* s[k]
			double rho_k = 0;
			size_t m = 0;
			for (size_t n = 0; n < N; ++n) {
				if (phi_flags[n]) {
					r[m] += s_hat[k] * cos((M_PI / N) * (n + 0.5) * k) * factor;
					rho_k += r[m] * cos((M_PI / N) * (n + 0.5) * k) * factor;
					m++;
				}
			}

			double sign = (rho_k > 0 ? 1.0 : -1.0);
			s_hat[k] = (sign * max(fabs(rho_k) - lambda, 0)) / A_norm2[k];

			double dsk = fabs(s_hat[k] - s_k0);
			max_dsk = max(dsk, max_dsk);
			max_sk = max(fabs(s_hat[k]), max_sk);


			// Adding back the residual contribution from s_hat[k]
			m = 0;
			for (size_t n = 0; n < N; ++n) {
				if (phi_flags[n]) {
					r[m++] -= s_hat[k] * cos((M_PI / N) * (n + 0.5) * k) * factor;
				}
			}
		}
		if ((max_dsk / max_sk) < tol) break;
	}

	free(r);
	free(y_hat);
	free(A_norm2);
}

void get_random_sample_flags_from_file(char *filepath, size_t N, bool *flags) {
	FILE *fp = fopen(filepath, "r");
	if (fp) {
		int val;
		for (int i = 0; i < N; ++i) {
			// I'm afraid of int value clobbering many bools in one shot
			// somehow.
			// I don't want to bother checking that though, so val it is.
			fscanf(fp, "%d\n", &val);
			flags[i] = (bool) val;
		}
	}
	fclose(fp);
}

/*
void get_random_sample_flags(size_t N, size_t m, bool *flags) {
	for (int i = 0; i < N; ++i) {
		flags[i] = false;
	}

	if (m > N) m = N;
	int n_flags = 0;
	while (n_flags < m) {
		size_t idx = rand() % N;
		if (!flags[idx]) {
			flags[idx] = true;
			n_flags++;
		}
	}
}
*/

void get_selected_elements(double *seq, bool *flags, int N, int m, double *out) {
	size_t idx = 0;
	for (int i = 0; i < N; ++i) {
		if (flags[i]) {
			out[idx++] = seq[i];
		}
		if (idx == m) break;
	}
}

void ppg(double *t0, double *X0, size_t N0,
		 bool *phi_flags, const PPG_Params params,
		 double *Xr) {
	/*
	 * t0		:	sample instants (in s) 
	 * X0		:	original samples (which we are trying to reconstruct)
	 * N0		:	number of samples we have
	 * phi_flags:	boolean flags to denote which observations we are selecting,
	 *				this should have been randomly generated previously
	 * params	:	simulation/experiment parameters
	 * Xr		:	where we should store the reconstructed signal (output)
	 */
	
	size_t N_window = params.N_window;
	size_t M = params.M;

	double *y = malloc(sizeof(double) * M);
	double *s = malloc(sizeof(double) * N_window);
	double *xr = malloc(sizeof(double) * N_window);

	for (size_t t = 0; t < N0 - N_window + 1; t += N_window / 2) {
		get_selected_elements(X0 + t, phi_flags, N_window, M, y);
		cd_lasso_randsampleDCT(y, phi_flags, M, N_window, 0.01, s, 1.0e-2);
		idct_1D(s, N_window, xr);
		memcpy(Xr + t + N_window / 4,
			   xr	  + N_window / 4,
			   sizeof(double) * N_window / 2);
	}
	
	free(xr);
	free(s);
	free(y);
}

/*
 * argv[0]
 * argv[1]: samples.csv
 * argv[2]: reconstructed.csv
 * argv[3]: phi_flags.csv
 */
int main(int argc, char **argv) {
	if (argc < 4) {
		fprintf(stderr, "Not enough arguments!\n");
		fprintf(stderr, "argv[1]: samples.csv\n");
		fprintf(stderr, "argv[2]: reconstructed.csv\n");
		fprintf(stderr, "argv[3]: phi_flags.csv\n");
	}

	srand(time(NULL));
	PPG_Params params = get_ppg_params();
	printf("f0: %.1lf\nT0: %.3lf\n", params.f0, params.T0);
	printf("N_window: %ld\nM: %ld\n", params.N_window, params.M);

	const size_t MAX_N = params.MAX_SAMPLES;
	size_t N0 = 0;
	double t0[MAX_N], X0[MAX_N];
	double Xr[MAX_N];

	FILE* fp_samples = fopen(argv[1], "r");
	if (!fp_samples) {
		fprintf(stderr, "Couldn't open file %s!\n", argv[1]);
		exit(EXIT_FAILURE);
	} else {
		while (fscanf(fp_samples, "%lf,%lf\n", &t0[N0], &X0[N0]) == 2) {
			N0++;
			if (N0 == MAX_N) break;
		}
		fclose(fp_samples);
	}

	printf("Read %ld samples.\n", N0);

	size_t N_window = params.N_window;
	bool *phi_flags = malloc(sizeof(bool) * N_window);
	get_random_sample_flags_from_file(argv[3], N_window, phi_flags);
	// get_random_sample_flags(N_window, M, phi_flags);

	// Just making things simpler and removing the chance of incomplete windows.
	N0 = NEAREST_MULTIPLE(N0, N_window / 2);
	printf("N0: %ld\n", N0);

	// call ppg algorithm
	ppg(t0, X0, N0, phi_flags, params, Xr);

	printf("Writing to file %s...\n", argv[2]);
	FILE *fp_write = fopen(argv[2], "w");
	if (!fp_write) {
		fprintf(stderr, "Couldn't open file %s for writing!\n", argv[2]);
	} else {
		for (size_t i = 0; i < N0; ++i) {
			fprintf(fp_write, "%f,%f\n", t0[i], Xr[i]);
		}
	}
	fclose(fp_write);

	free(phi_flags);
	return 0;
}
