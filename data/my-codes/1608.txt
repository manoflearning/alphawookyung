#include <iostream>
using namespace std;

int main() {
	int t;
	scanf("%d", &t);

	for (int test = 0; test < t; test++) {
		int a, b;
		scanf("%d %d", &a, &b);

		int ans = 0;
		if (1 <= a && a <= 21) {
			if (a <= 1) ans += 5000000;
			else if (a <= 3) ans += 3000000;
			else if (a <= 6) ans += 2000000;
			else if (a <= 10) ans += 500000;
			else if (a <= 15) ans += 300000;
			else if (a <= 21) ans += 100000;
		}
		if (1 <= b && b <= 31) {
			if (b <= 1) ans += 5120000;
			else if (b <= 3) ans += 2560000;
			else if (b <= 7) ans += 1280000;
			else if (b <= 15) ans += 640000;
			else if (b <= 31) ans += 320000;
		}

		printf("%d\n", ans);
	}

	return 0;
}