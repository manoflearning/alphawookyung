#include <iostream>
using namespace std;

int main() {
	int n;
	scanf("%d", &n);

	int c = 100, s = 100;

	for (int i = 0; i < n; i++) {
		int d1, d2;
		scanf("%d %d", &d1, &d2);

		if (d1 > d2) s -= d1;
		else if (d1 < d2) c -= d2;
	}

	printf("%d\n%d", c, s);

	return 0;
}