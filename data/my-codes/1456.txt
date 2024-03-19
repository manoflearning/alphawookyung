#include <stdio.h>
int i, l;

int main (void) {
	int input[10];
	int flag;
	float count=0;
	
	for(i=0; i<10; i++) {
		scanf("%d", &input[i]);
		input[i] = input[i]%42;	
	}
		
	for(i=0; i<10; i++) {
		flag=1;
		for(l=0; l<10; l++)
			if(input[l]==input[i] && i!=l)
				flag++;
		
		count += (float)1/flag;
	}
	
	printf("%.0f", count);
	
	return 0;
}