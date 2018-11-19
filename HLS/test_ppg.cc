/*
 * argv[0]
 * argv[1]: samples.csv
 * argv[2]: reconstructed.csv
 * argv[3]: phi_flags.csv
 */
#include "ppg.hh"

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
	ppg( t0, X0, Xr);

	FILE *fp_write = fopen(argv[2], "w");
	if (fp_write) {
		for (size_t i = 0; i < N0; ++i) {
			fprintf(fp_write, "%f,%f\n", t0[i], Xr[i]);
		}
	}
	fclose(fp_write);
	return 0;
}

void get_random_sample_flags_from_file(char *filepath, size_t N, bool *flags) {
#pragma HLS PIPELINE
	FILE *fp = fopen(filepath, "r");
	if (fp) {
		int val;
		for (int i = 0; i < N; ++i) {
#pragma HLS UNROLL
			// I'm afraid of int value clobbering many bools in one shot
			// somehow.
			// I don't want to bother checking that though, so val it is.
			fscanf(fp, "%d\n", &val);
			flags[i] = (bool) val;
		}
	}
	fclose(fp);
}
