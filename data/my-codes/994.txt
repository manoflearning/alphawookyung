#include <stdio.h>

int i, l, j;

void sangsu_comp(char *A, char *B) {
	for(i=2; i>=0; i--) {
		if(A[i]>B[i]) {
			printf("%c%c%c", A[2], A[1], A[0]);
			break;
		}
		else if(B[i]>A[i]) {
			printf("%c%c%c", B[2], B[1], B[0]);
			break;
		}
		else continue;
	}
}

int main (void) {
	char A[4], B[4];
	
	scanf("%s %s", A, B);
	
	sangsu_comp(A, B);
	
	return 0;
}