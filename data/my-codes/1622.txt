#include <stdio.h>

int main(void) {
	int min=1001, max=-1;
	int sam;
	int n;
	int i;
	
	scanf("%d", &n);
	
	for(i=0; i<n; i++) {
		scanf("%d", &sam);
		
		if(sam < min) min=sam;
		if(sam > max) max=sam;
	}
	
	printf("%d", max-min);
	
	return 0;
}