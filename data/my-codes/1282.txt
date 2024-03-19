#include <iostream>
using namespace std;

int main() {
	int t;
	scanf("%d", &t);

	for (int test = 0; test < t; test++) {
		int yon = 0, kor = 0;

		for (int i = 0; i < 9; i++) {
			int s1, s2;
			scanf("%d %d", &s1, &s2);

			yon += s1, kor += s2;
		}

		if (yon > kor) printf("Yonsei\n");
		else if (yon < kor) printf("Korea\n");
		else printf("Draw\n");
	}

	return 0;
}