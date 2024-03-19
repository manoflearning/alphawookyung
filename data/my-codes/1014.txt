#include <stdio.h>

int least_common_multiple(int a, int b);
int greatest_common_measure(int a, int b);

int main(void) {
	int a, b;
	int t;
	int i;
	
	scanf("%d", &t);
	
	for(i=0; i<t; i++) {
		scanf("%d %d", &a, &b);
	
		printf("%d %d\n", least_common_multiple(a, b), greatest_common_measure(a, b));
	}
	return 0;
}

int least_common_multiple(int a, int b) {
	int answer;
	
	if(a>=b) {
		answer=a;
		while(1) {
			if(answer%b==0) return answer;
			
			answer+=a;
		}
	}
	else {
		answer=b;
		while(1) {
			if(answer%a==0) return answer;
			
			answer+=b;
		}
	}
}
int greatest_common_measure(int a, int b) {
	int i;
	
	if(a>=b) {
		for(i=1; i<=b; i++)
			if(b%i==0 && a%(b/i)==0) return b/i;
		}
	else
		for(i=1; i<=a; i++)
			if(a%i==0 && b%(a/i)==0)
				return a/i;
}