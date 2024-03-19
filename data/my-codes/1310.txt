#include <stdio.h>

int main(void) {
	int n;
	int stu, app;
	int count=0;
	int i;
	
	scanf("%d", &n);
	
	for(i=0; i<n; i++) {
		scanf("%d %d", &stu, &app);
		
		count+=app%stu;
	}
	
	printf("%d", count);
	
	return 0;
}