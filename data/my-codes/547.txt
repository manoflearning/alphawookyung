#include <stdio.h>
#include <math.h>

int elementary_school_mathematics(int a, int b);

int main(void) {
	int t;
	int a, b;
	int answer;
	int i;
	
	scanf("%d", &t);
	
	for(i=0; i<t; i++) {
		scanf("%d %d", &a, &b);
		
		answer = elementary_school_mathematics(a%10, b);
	
		if(answer!=0) printf("%d\n", answer);
		else printf("10\n");
	}
	
	return 0;
}

int elementary_school_mathematics(int a, int b) {
	if(a==1 || a==5 || a==6) return a;
	else if(a==4 || a==9)
		if(b%2==1) return a;
		else return (int)pow(a, 2)%10;
	else
		if(b%4==1) return a;
		else if(b%4==2) return (int)pow(a, 2)%10;
		else if(b%4==3) return (int)pow(a, 3)%10;
		else if(b%4==0) return (int)pow(a, 4)%10;
}
