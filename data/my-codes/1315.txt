#include <iostream>
using namespace std;

int main() {
	int t;
	scanf("%d", &t);

	for (int test = 0; test < t; test++) {
		int n;
		scanf("%d", &n);

		double sum_c = 0, sum_g = 0;
		for (int i = 0; i < n; i++) {
			double c, g;
			scanf("%lf %lf", &c, &g);

			sum_c += c;
			sum_g += c * g;
		}

		printf("%.0lf %.1lf\n", sum_c, sum_g / sum_c);
	}

	return 0;
}