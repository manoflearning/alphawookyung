#include <stdio.h>

void fibonacci(int f1[], int f2[]);

int main(void) {
	int t, n;
	int f1[41] = {1, 0}, f2[41] = {0, 1};
	int i;
	
	fibonacci(f1, f2);
	
	scanf("%d", &t);
	
	for(i=0; i<t; i++) {
		scanf("%d", &n);
		
		printf("%d %d\n", f1[n], f2[n]);
	}
	
	return 0;
}

void fibonacci(int f1[], int f2[]) {
    int i;
    
	for(i=2; i<=40; i++) {
		f1[i] = f1[i-1] + f1[i-2];
		f2[i] = f2[i-1] + f2[i-2];
	}
}