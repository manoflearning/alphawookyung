#include <stdio.h>

double u_circle(int R) {
	return R*R*3.14159265358979323846;
}

double taxi_circle(int R) {
	return R*R*2;
}

int main (void) {
	int R;
	
	scanf("%d", &R);
	
	printf("%.7f\n%.7f", u_circle(R), taxi_circle(R));
	
	return 0;
}