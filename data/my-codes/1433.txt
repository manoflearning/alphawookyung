#include <iostream>
using namespace std;

int main() {
	int t;
	scanf("%d", &t);

	for (int test = 0; test < t; test++) {
		int min = 102;
		int sum = 0;

		for (int i = 0; i < 7; i++) {
			int sam;
			scanf("%d", &sam);

			if (sam % 2 == 0) {
				sum += sam;
				if (sam < min) min = sam;
			}
		}

		printf("%d %d\n", sum, min);
	}

	return 0;
}