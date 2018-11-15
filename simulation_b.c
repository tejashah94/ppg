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
double y[20];
double s[240];
double xr[240];
double A_norm2[240];
double y_hat[20];
double r[20];

void ppg() {
	for (size_t t = 0; t < 10000 - 240 + 1; t += 240 / 2) {
		// get_selected_elements(X0 + t, phi_flags, N_window, M, y);
		size_t idx = 0;
		for (size_t i = 0; i < 240; ++i) {
			if (phi_flags[i]) {
				y[idx++] = X0[t + i];
			}
			if (idx == M) break;
		}

		// cd_lasso_randsampleDCT(y, phi_flags, M, N_window, 0.01, s, 1.0e-2);
		//
		// calc_Anorm2_randsampleDCT(phi_flags, N, A_norm2);
		memset(A_norm2, 0, sizeof(double) * 240);
		const double factor_1 = 2.0 / sqrt(240);
		const double factor_0 = factor_1 / sqrt(2);
		for (size_t n = 0; n < 240; ++n) {
			if (phi_flags[n]) {
				A_norm2[0] += factor_0 * factor_0;
				for (size_t k = 1; k < 240; ++k) {
					A_norm2[k] += pow(cos((M_PI / 240) * (n + 0.5) * k) * factor_1, 2);
				}
			}
		}

		for (size_t k = 0; k < 240; ++k) {
			s[k] = 0.5;
		}

		// random_sample_idct_1D(s, N, phi_flags, M, y_hat);
		memset(y_hat, 0, sizeof(double) * 20);
		for (size_t m = 0; m < 20; ++m) {
			y_hat[m] = s[0] * factor_0;
		}

		size_t m = 0;
		for (size_t n = 0; n < 240; ++n) {
			if (phi_flags[n]) {
				for (size_t k = 1; k < 240; ++k) {
					y_hat[m] += cos((M_PI / 240) * (n + 0.5) * k) * s[k] * factor_1;
				}
				m++;
			}
		}

		memcpy(r, y, sizeof(double) * 20);
		// vec_sub_update(r, y_hat, M);
		for (size_t i = 0; i < 20; ++i) {
			r[i] -= y_hat[i];
		}

		double max_sk = 0.5; // depends on initialization
		while (true) {
			double max_dsk = 0;
			for (size_t k = 0; k < 240; ++k) {
				if (A_norm2[k] == 0.0)
					continue;

				double factor = (k == 0 ? factor_0 : factor_1);
				double s_k0 = s[k];

				// r += A[:,k] .* s[k]
				double rho_k = 0;
				size_t m = 0;
				for (size_t n = 0; n < 240; ++n) {
					if (phi_flags[n]) {
						r[m] += s[k] * cos((M_PI / 240) * (n + 0.5) * k) * factor;
						rho_k += r[m] * cos((M_PI / 240) * (n + 0.5) * k) * factor;
						m++;
					}
				}

				double sign = (rho_k > 0 ? 1.0 : -1.0);
				s[k] = (sign * MAX(fabs(rho_k) - LASSO_lambda, 0)) / A_norm2[k];

				double dsk = fabs(s[k] - s_k0);
				max_dsk = MAX(dsk, max_dsk);
				max_sk = MAX(fabs(s[k]), max_sk);

				// Adding back the residual contribution from s_hat[k]
				m = 0;
				for (size_t n = 0; n < 240; ++n) {
					if (phi_flags[n]) {
						r[m++] -= s[k] * cos((M_PI / 240) * (n + 0.5) * k) * factor;
					}
				}
			}
			if ((max_dsk / max_sk) < CD_diff_thresh) break;
		}

		// idct_1D(s, N_window, xr);
		memset(xr, 0, sizeof(double) * 240);
		for (size_t n = 0; n < 240; ++n) {
			xr[n] += s[0] * factor_0;
		}

		for (size_t k = 1; k < 240; ++k) {
			for (size_t n = 0; n < 240; ++n) {
				xr[n] += cos((M_PI / 240) * (n + 0.5) * k) * s[k] * factor_1;
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
