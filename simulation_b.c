/*
 * A single-block version of simulation.c with all array sizes being compile
 * time constants. Hopefully this way they can be synthesized in hardware.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#define MAX(a, b) (a > b ? a : b)
#define NEAREST_MULTIPLE(n, d) (n - n % d)

// Parameters
const int MAX_SAMPLES = 10000;
const double f0 = 4.0;
const double T0 = 1.0 / 4.0;
const double T_window = 60;
const size_t N_window = 240;
const unsigned int CF = 12;
const size_t M = 20;
const double LASSO_lambda = 1.0e-2;
const double CD_diff_thresh = 1.0e-2;

// Arrays needed
double t0[10000];
double X0[10000];
double Xr[10000];
bool phi_flags[240];
double psi[240 * 240];
double psiT[240 * 240];
double A[20 * 240];
double y[20];
double s[240];
double xr[240];
double A_norm2[240];
double y_hat[20];
double r[20];

void ppg() {
	// DCT basis matrix
	// TODO: implement fast DCT so this isn't needed.
	for (int k = 0; k < 240; ++k) {
		double factor = (2.0 / sqrt(240)) * (k == 0 ? 1.0 / sqrt(2) : 1.0);
		for (int n = 0; n < 240; ++n) {
			psi[k * 240 + n] = factor * cos((M_PI / 240) * (n + 0.5) * k);
			psiT[n * 240 + k] = psi[k * 240 + n];
		}
	}

	// get_selected_rows(psiT, phi_flags, N_window, M, A);
	size_t row = 0;
	for (size_t i = 0; i < 240; ++i) {
		if (phi_flags[i]) {
			memcpy(A + row * 240, psiT + i * 240, 240 * sizeof(double));
			if (row++ == 20) break;
		}
	}

	for (size_t t = 0; t < 10000 - 240 + 1; t += 240 / 2) {
		// get_selected_elements(X0 + t, phi_flags, N_window, M, y);
		size_t idx = 0;
		for (size_t i = 0; i < 240; ++i) {
			if (phi_flags[i]) {
				y[idx++] = X0[t + i];
			}
			if (idx == 20) break;
		}
		
		// cd_lasso(y, A, M, N_window, 0.01, s, 1.0e-2);
		// y M-dim, A is MxN, s is Nx1
		memset(A_norm2, 0, sizeof(double) * 240);
		for (size_t i = 0; i < 20; ++i) {
			for (size_t j = 0; j < 240; ++j) {
				A_norm2[j] += pow(A[i * 240 + j], 2);
			}
		}

		for (size_t i = 0; i < 240; ++i) {
			s[i] = 0.5;
		}

		// y_hat = A * s
		// dot(A, s, M, N, 1, y_hat);
		for (size_t i = 0; i < 20; ++i) {
			y_hat[i] = 0;
			for (size_t j = 0; j < 240; ++j) {
				y_hat[i] += A[i * 240 + j] * s[j];
			}
		}

		memcpy(r, y, sizeof(double) * 20);
		for (size_t i = 0; i < 20; ++i) {
			r[i] -= y_hat[i];
		}

		double max_si = 0;
		for (size_t i = 0; i < 240; ++i) {
			if (fabs(s[i]) > max_si) {
				max_si = fabs(s[i]);
			}
		}

		while (true) {
			double max_dsi = 0;
			for (size_t i = 0; i < 240; ++i) {
				if (A_norm2[i] == 0.0) {
					continue;
				}

				double s_i0 = s[i];

				// r += A[:,i] .* x[i]
				double rho_i = 0;
				for (size_t j = 0; j < 20; ++j) {
					r[j] += A[j * 240 + i] * s[i];
					rho_i += r[j] * A[j * 240 + i];
				}

				double sign = (rho_i > 0 ? 1.0 : -1.0);
				s[i] = (sign * MAX(fabs(rho_i) - LASSO_lambda, 0)) / A_norm2[i];

				double dsi = fabs(s[i] - s_i0);
				max_dsi = (dsi > max_dsi ? dsi : max_dsi);

				for (int j = 0; j < 20; ++j) {
					r[j] -= s[i] * A[j * 240 + i];
				}

				max_si = MAX(max_si, fabs(s[i]));
			}
			if ((max_dsi / max_si) < CD_diff_thresh) {
				break;
			}
		}

		// dot(psiT, s, N_window, N_window, 1, xr);
		for (size_t i = 0; i < 240; ++i) {
			xr[i] = 0;
			for (size_t j = 0; j < 240; ++j) {
				xr[i] += psiT[i * 240 + j] * s[j];
			}
		}

		memcpy(Xr + t + 240 / 4,
			   xr	  + 240 / 4,
			   sizeof(double) * 240 / 2);
	}
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
	size_t N0 = 0;
	FILE* fp_samples = fopen(argv[1], "r");
	if (fp_samples) {
		while (fscanf(fp_samples, "%lf,%lf\n", &t0[N0], &X0[N0]) == 2) {
			N0++;
			if (N0 == 10000) break;
		}
		fclose(fp_samples);
	}

	N0 = NEAREST_MULTIPLE(N0, N_window / 2);
	get_random_sample_flags_from_file(argv[3], N_window, phi_flags);
	memset(Xr, 0, sizeof(double) * 10000);
	ppg();

	FILE *fp_write = fopen(argv[2], "w");
	if (fp_write) {
		for (size_t i = 0; i < N0; ++i) {
			fprintf(fp_write, "%f,%f\n", t0[i], Xr[i]);
		}
	}
	fclose(fp_write);
	return 0;
}
