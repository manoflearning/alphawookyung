#include <iostream>
using namespace std;

int main() {
	int n;
	scanf("%d", &n);

	while (true) {
		int flag;
		scanf("%d", &flag);

		if (flag == 0) break;

		if (flag % n != 0) printf("%d is NOT a multiple of %d.\n", flag, n);
		else printf("%d is a multiple of %d.\n", flag, n);
	}

	return 0;
}