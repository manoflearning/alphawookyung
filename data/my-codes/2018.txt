#include <iostream>
using namespace std;

int main() {
	int t;
	scanf("%d", &t);

	for (int test = 0; test < t; test++) {
		char sentence[31];
		scanf(" %[^\n]s", sentence);

		if ('a' <= sentence[0] && sentence[0] <= 'z')
			sentence[0] += 'A' - 'a';

		printf("%s\n", sentence);
	}

	return 0;
}