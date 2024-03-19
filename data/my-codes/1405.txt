#include <iostream>
#include <cstring>
using namespace std;

int main() {
	char mixed[51];
	scanf("%s", mixed);

	int flag = 1;

	int len = strlen(mixed);
	for (int i = 0; i < len; i++) {
		if (mixed[i] == 'A') {
			if (flag == 1) flag = 2;
			else if (flag == 2) flag = 1;
		}
		else if (mixed[i] == 'B') {
			if (flag == 2) flag = 3;
			else if (flag == 3) flag = 2;
		}
		else {
			if (flag == 1) flag = 3;
			else if (flag == 3) flag = 1;
		}
	}

	printf("%d", flag);

	return 0;
}