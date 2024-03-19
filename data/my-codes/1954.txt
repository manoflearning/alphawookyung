#include <stdio.h>

int main(void) {
	int n[31] = {1,};
	int sample;
	int i;
	
	for(i=0; i<28; i++) {
		scanf("%d", &sample);
		
		n[sample] = 1;
	}
	
	for(i=0; i<31; i++)
		if(n[i] == 0)
			printf("%d\n", i);
	
	return 0;
}