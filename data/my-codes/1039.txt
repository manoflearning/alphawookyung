#include <iostream>
#include <algorithm>
using namespace std;

int main() {
	int t;
	scanf("%d", &t);

	int ans = 0;

	for (int test = 0; test < t; test++) {
		int a, b, c;
		scanf("%d %d %d", &a, &b, &c);

		if (a == b && b == c) ans = max(ans, 10000 + a * 1000);
		else if (a == b) ans = max(ans, 1000 + a * 100);
		else if (a == c) ans = max(ans, 1000 + a * 100);
		else if (b == c) ans = max(ans, 1000 + b * 100);
		else {
			int max_dice = 0;
			max_dice = max(a, max_dice);
			max_dice = max(b, max_dice);
			max_dice = max(c, max_dice);

			ans = max(ans, max_dice * 100);
		}
	}

	printf("%d", ans);

	return 0;
}