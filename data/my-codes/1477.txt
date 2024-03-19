#include <iostream>
using namespace std;

int main() {
	int n, w, h;
	scanf("%d %d %d", &n, &w, &h);

	int checker = w * w + h * h;

	for (int test = 0; test < n; test++) {
		int stick;
		scanf("%d", &stick);

		if (stick * stick <= checker) printf("DA\n");
		else printf("NE\n");
	}

	return 0;
}