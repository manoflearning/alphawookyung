#include <iostream>
using namespace std;

int main() {
	int n;
	scanf("%d", &n);

	bool ans = false;
	for (long long i = 1; i <= n; i *= 2)
		if (i == n) ans = true;

	printf("%d", ans);

	return 0;
}