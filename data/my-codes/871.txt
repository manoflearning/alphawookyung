#include <iostream>
#include <cstring>
using namespace std;

int main() {
	char score[3];
	scanf("%s", score);

	if (strcmp("A+", score) == 0) printf("4.3");
	else if (strcmp("A0", score) == 0) printf("4.0");
	else if (strcmp("A-", score) == 0) printf("3.7");
	else if (strcmp("B+", score) == 0) printf("3.3");
	else if (strcmp("B0", score) == 0) printf("3.0");
	else if (strcmp("B-", score) == 0) printf("2.7");
	else if (strcmp("C+", score) == 0) printf("2.3");
	else if (strcmp("C0", score) == 0) printf("2.0");
	else if (strcmp("C-", score) == 0) printf("1.7");
	else if (strcmp("D+", score) == 0) printf("1.3");
	else if (strcmp("D0", score) == 0) printf("1.0");
	else if (strcmp("D-", score) == 0) printf("0.7");
	else if (strcmp("F", score) == 0) printf("0.0");

	return 0;
}