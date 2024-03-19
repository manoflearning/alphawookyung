#include <stdio.h>

void factor_multiple(int a, int b) {
	int flag;
	
	if(a>b) {
		flag=b;
		while(1) {
			flag+=b;
			if(a==flag) {
				printf("multiple\n");
				break;
			}
			if(a<flag) {
				printf("neither\n");
				break;
			}
		}
	}
	else if(a<b) {
		flag=a;
		while(1) {
			flag+=a;
			if(b==flag) {
				printf("factor\n");
				break;
			}
			if(b<flag) {
				printf("neither\n");
				break;
			}
		}
	}
}

int main (void) {
	int a, b;
	
	while(1) {
		scanf("%d %d", &a, &b);
		
		if(a==0 && b==0) break;
		
		factor_multiple(a, b);
	}
	
	return 0;
}