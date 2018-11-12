/**
 * Algorithm for PPG signal reconstruction from compressively sampled input.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

// #include "setup.h"
#define NEAREST_MULTIPLE(n, d) (n - n % d)

const size_t MAX_SAMPLES = 10000;	// maximum number of samples we can handle

const double f0 = 4.0;				// sample rate of original signal, in Hz
const double T0 = 1.0 / 4.0;		// 1 / f0, so in seconds

const double T_window = 60;			// Window length for reconstruction algorithm,
									// in seconds
const size_t N_window = 240;		// #samples in that window
									//  = NEAREST_MULTIPLE(T_window / T0, 4)

const unsigned int CF = 12;			// compression factor
const size_t M = 20;				// compressed window length = N_window / CF

const double LASSO_lambda = 0.01;	// regularization parameter for LASSO
const double CD_diff_thresh = 0.01;	// stopping threshold for coordinate descent
									// algorithm

// from lasso
// double A_norm2[N_window];
double A_norm2[240];
// double y_hat[M];
double y_hat[20];
// double r[M];
double r[20];

// from ppg
// double psi[N_window * N_window];
double psi[240 * 240];
// double psiT[N_window * N_window];
double psiT[240 * 240];
// double A[M * N_window];
double A[20 * 240];
// double y[M];
double y[20];
// double s[N_window];
double s[240];
// double xr[N_window];
double xr[240];

// from main
// bool phi_flags[N_window];
bool phi_flags[240];

double max(double a, double b) {
	return (a > b ? a : b) ;
}

void dot(double *A, double *B, size_t M, size_t N, size_t P, double *C) {
	// C = AB, everything is row-major
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < P; ++j) {
			C[i * P + j] = 0;
			for (int k = 0; k < N; ++k) {
				C[i * P + j] += A[i * N + k] * B[k * P + j];
			}
		}
	}
}

void vecsub(double *a, double *b, size_t N, double *out) {
	for (size_t i = 0; i < N; ++i) {
		out[i] = a[i] - b[i];
	}
}

void cd_lasso(double *y, double *A,
				size_t M, size_t N, double lambda,
				double *x_hat, double tol) {
	// y M-dim, A is MxN, x_hat is Nx1
	// double *A_norm2 = malloc(sizeof(double) * N);
	// double A_norm2[N];
	memset(A_norm2, 0, sizeof(double) * N);
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			A_norm2[j] += pow(A[i * N + j], 2);
		}
	}

	for (int i = 0; i < N; ++i) {
		x_hat[i] = 0.5;
	}

	// double *y_hat = malloc(sizeof(double) * M);
	// y_hat = A * x_hat
	dot(A, x_hat, M, N, 1, y_hat);

	// double *r = malloc(sizeof(double) * M);
	memcpy(r, y, sizeof(double) * M);
	vecsub(y, y_hat, M, r);

	double max_xi = 0;
	for (int i = 0; i < N; ++i) {
		if (fabs(x_hat[i]) > max_xi) {
			max_xi = fabs(x_hat[i]);
		}
	}

	while (true) {
		double max_dxi = 0;
		for (int i = 0; i < N; ++i) {
			if (A_norm2[i] == 0.0) {
				continue;
			}

			double x_i0 = x_hat[i];

			// r += A[:,i] .* x[i]
			double rho_i = 0;
			for (int j = 0; j < M; ++j) {
				r[j] += A[j * N + i] * x_hat[i];
				rho_i += r[j] * A[j * N + i];
			}

			double sign = (rho_i > 0 ? 1.0 : -1.0);
			x_hat[i] = (sign * max(fabs(rho_i) - lambda, 0)) / A_norm2[i];

			double dxi = fabs(x_hat[i] - x_i0);
			max_dxi = (dxi > max_dxi ? dxi : max_dxi);

			for (int j = 0; j < M; ++j) {
				r[j] -= x_hat[i] * A[j * N + i];
			}

			max_xi = max(max_xi, x_hat[i]);
		}
		if ((max_dxi / max_xi) < tol) {
			break;
		}
	}

	// free(r);
	// free(y_hat);
	// free(A_norm2);
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

void get_selected_elements(double *seq, bool *flags, int N, int m, double *out) {
	size_t idx = 0;
	for (int i = 0; i < N; ++i) {
		if (flags[i]) {
			out[idx++] = seq[i];
		}
		if (idx == m) break;
	}
}

void get_selected_rows(double *mat, bool *flags, int N, int m, double *out) {
	size_t row = 0;
	for (int i = 0; i < N; ++i) {
		if (flags[i]) {
			memcpy(out + row * N, mat + i * N, N * sizeof(double));
			if (row++ == m) break;
		}
	}
}

/*
void ppg(double *t0, double *X0, size_t N0,
		 bool *phi_flags, const PPG_Params params,
		 double *Xr) {
*/
void ppg(double *t0, double *X0, size_t N0,
		 bool *phi_flags, double *Xr) {
	/*
	 * t0		:	sample instants (in s) 
	 * X0		:	original samples (which we are trying to reconstruct)
	 * N0		:	number of samples we have
	 * phi_flags:	boolean flags to denote which observations we are selecting,
	 *				this should have been randomly generated previously
	 * params	:	simulation/experiment parameters
	 * Xr		:	where we should store the reconstructed signal (output)
	 */
	
	// size_t N_window = params.N_window;
	// size_t M = params.M;

	// DCT basis matrix
	// TODO: implement fast DCT so this isn't needed.
	// double *psi = malloc(sizeof(double) * N_window * N_window);
	// This is psi.T. Only keeping a separate copy because I am lazy.
	// TODO: become less lazy. Also write something better.
	// double *psiT = malloc(sizeof(double) * N_window * N_window);
	for (int k = 0; k < N_window; ++k) {
		double factor = (2.0 / sqrt(N_window)) * (k == 0 ? 1.0 / sqrt(2) : 1.0);
		for (int n = 0; n < N_window; ++n) {
			psi[k * N_window + n] = factor * cos((M_PI / N_window) * (n + 0.5) * k);
			psiT[n * N_window + k] = psi[k * N_window + n];
		}
	}

	// double *A = malloc(sizeof(double) * M * N_window);
	get_selected_rows(psiT, phi_flags, N_window, M, A);

	// double *y = malloc(sizeof(double) * M);
	// double *s = malloc(sizeof(double) * N_window);
	// double *xr = malloc(sizeof(double) * N_window);

	for (size_t t = 0; t < N0 - N_window + 1; t += N_window / 2) {
		get_selected_elements(X0 + t, phi_flags, N_window, M, y);
		cd_lasso(y, A, M, N_window, 0.01, s, 1.0e-2);
		dot(psiT, s, N_window, N_window, 1, xr);
		memcpy(Xr + t + N_window / 4,
			   xr	  + N_window / 4,
			   sizeof(double) * N_window / 2);
	}
	
	// free(xr);
	// free(s);
	// free(y);
	// free(A);
	// free(psiT);
	// free(psi);
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
	/*
	PPG_Params params = get_ppg_params();
	printf("f0: %.1lf\nT0: %.3lf\n", params.f0, params.T0);
	printf("N_window: %ld\nM: %ld\n", params.N_window, params.M);
	*/

	// const size_t MAX_N = params.MAX_SAMPLES;
	const size_t MAX_N = MAX_SAMPLES;
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

	// size_t N_window = params.N_window;
	// bool *phi_flags = malloc(sizeof(bool) * N_window);
	get_random_sample_flags_from_file(argv[3], N_window, phi_flags);
	// get_random_sample_flags(N_window, M, phi_flags);

	// Just making things simpler and removing the chance of incomplete windows.
	N0 = NEAREST_MULTIPLE(N0, N_window / 2);
	printf("N0: %ld\n", N0);

	// call ppg algorithm
	// ppg(t0, X0, N0, phi_flags, params, Xr);
	ppg(t0, X0, N0, phi_flags, Xr);

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

	// free(phi_flags);
	return 0;
}
