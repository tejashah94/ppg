/**
 * Algorithm for PPG signal reconstruction from compressively sampled input.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

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

void vecadd(double *a, double *b, size_t N, bool sub_flag, double *out) {
	if (sub_flag) {
		for (size_t i = 0; i < N; ++i) {
			out[i] = a[i] - b[i];
		}
	} else {
		for (size_t i = 0; i < N; ++i) {
			out[i] = a[i] + b[i];
		}
	}
}

void cd_lasso(double *y, double *A,
				size_t M, size_t N, double lambda,
				double *x_hat, double tol) {
	// y M-dim, A is MxN, x_hat is Nx1
	double *A_norm2 = malloc(sizeof(double) * N);
	memset(A_norm2, 0, sizeof(double) * N);
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			A_norm2[j] += pow(A[i * N + j], 2);
		}
	}

	for (int i = 0; i < N; ++i) {
		x_hat[i] = 0.5;
	}

	double *y_hat = malloc(sizeof(double) * M);
	// y_hat = A * x_hat
	dot(A, x_hat, M, N, 1, y_hat);

	double *r = malloc(sizeof(double) * M);
	memcpy(r, y, sizeof(double) * M);
	vecadd(y, y_hat, M, true, r);

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

	const size_t MAX_N = 10000;
	size_t N0 = 0;
	double t0[MAX_N], X0[MAX_N];

	FILE* fp_samples = fopen(argv[1], "r");
	if (!fp_samples) {
		fprintf(stderr, "Couldn't open file %s!\n", argv[1]);
		exit(EXIT_FAILURE);
	} else {
		while (fscanf(fp_samples, "%lf,%lf\n", &t0[N0], &X0[N0]) == 2) {
			N0++;
		}
		fclose(fp_samples);
	}

	printf("Read %ld samples.\n", N0);

	const double f0 = 4.0;					// Hz
	const double T0 = 1.0 / f0;				// seconds

	const double T_window = 60;				// seconds
	size_t N_window = (size_t) floor(T_window / T0);
	// Window length must be a multiple of 4 for this method.
	if (N_window % 4 != 0)
		N_window -= N_window % 4;
	
	// Just making things simpler and removing the chance of incomplete windows.
	N0 -= N0 % (N_window / 2);

	// DCT basis matrix
	// TODO: implement fast DCT so this isn't needed.
	double *psi = malloc(sizeof(double) * N_window * N_window);
	double *psiT = malloc(sizeof(double) * N_window * N_window);
	for (int k = 0; k < N_window; ++k) {
		double factor = (2.0 / sqrt(N_window)) * (k == 0 ? 1.0 / sqrt(2) : 1.0);
		for (int n = 0; n < N_window; ++n) {
			psi[k * N_window + n] = factor * cos((M_PI / N_window) * (n + 0.5) * k);
			psiT[n * N_window + k] = psi[k * N_window + n];
		}
	}


	// Random sampling matrix
	const int CF = 12;						// compression factor
	size_t M = N_window / CF;
	
	printf("f0: %.1lf\nT0: %.3lf\n", f0, T0);
	printf("N0: %ld\nN_window: %ld\nM: %ld\n", N0, N_window, M);

	bool *phi_flags = malloc(sizeof(bool) * N_window);
	get_random_sample_flags_from_file(argv[3], N_window, phi_flags);
	// get_random_sample_flags(N_window, M, phi_flags);

	double *A = malloc(sizeof(double) * M * N_window);
	get_selected_rows(psiT, phi_flags, N_window, M, A);

	double Xr[MAX_N];
	double *y = malloc(sizeof(double) * M);
	double *s = malloc(sizeof(double) * N_window);
	double *xr = malloc(sizeof(double) * N_window);

	for (size_t t = 0; t < N0 - N_window + 1; t += N_window / 2) {
		get_selected_elements(X0 + t, phi_flags, N_window, M, y);
		cd_lasso(y, A, M, N_window, 0.01, s, 1.0e-2);
		dot(psiT, s, N_window, N_window, 1, xr);
		memcpy(Xr + t + N_window / 4,
			   xr	  + N_window / 4,
			   sizeof(double) * N_window / 2);
	}
	
	double corr_coef = 0.0;
	double X0_mag = 1.0, Xr_mag = 1.0;
	dot(Xr, X0, 1, N0, 1, &corr_coef);
	dot(Xr, Xr, 1, N0, 1, &Xr_mag);
	dot(X0, X0, 1, N0, 1, &X0_mag);
	Xr_mag = sqrt(Xr_mag);
	X0_mag = sqrt(X0_mag);
	corr_coef /= (Xr_mag * X0_mag);
	printf("Correlation coefficient: %lf\n", corr_coef); 
	
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

	free(xr);
	free(y);
	free(A);
	free(phi_flags);
	free(psiT);
	free(psi);
	return 0;
}
