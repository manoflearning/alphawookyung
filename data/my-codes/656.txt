#include <iostream>
using namespace std;

int main() {
	int t;
	scanf("%d", &t);

	for (int test = 0; test < t; test++) {
		int s, n;
		scanf("%d %d", &s, &n);

		int ans = s;
		for (int i = 0; i < n; i++) {
			int a, b;
			scanf("%d %d", &a, &b);

			ans += a * b;
		}

		printf("%d\n", ans);
	}

	return 0;
}