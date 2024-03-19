#include <stdio.h>

void is_Junhee_cute(int n);

int main(void) {
	int n;
	int i;
	
	scanf("%d", &n);
	
	is_Junhee_cute(n);
	
	return 0;
}

void is_Junhee_cute(int n) {
	int s, no_cute=0, cute=0;
	int i;
	
	for(i=0; i<n; i++) {
		scanf("%d", &s);
		
		if(s==0) no_cute++;
		else cute++;
	}
	
	if(no_cute>cute) printf("Junhee is not cute!");
	else printf("Junhee is cute!");
}