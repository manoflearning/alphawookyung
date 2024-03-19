#include <stdio.h>
#include <stdlib.h>

int maximum_weight(int n);
int compare (void *first, void *second);

int main(void) {
	int n;
	
	scanf("%d", &n);
	
	printf("%d", maximum_weight(n));
	
	return 0;
}

int maximum_weight(int n) {
	int weight[99999];
	int max=0;
	int i;
	
	for(i=0; i<n; i++)
		scanf("%d", &weight[i]);
		
	qsort(weight, n, sizeof(weight[0]), compare);
	
	for(i=0; i<n; i++) {
		if(weight[i]*(n-i)>max) max=weight[i]*(n-i);
	}
	
	return max;
}

int compare (void *first, void *second)
{
    if (*(int*)first > *(int*)second)
        return 1;
    else if (*(int*)first < *(int*)second)
        return -1;
    else 
        return 0;
}
