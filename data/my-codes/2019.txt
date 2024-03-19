#include <stdio.h>
#include <math.h>

int i, l, j;

void selfnum_count(int *selfnum, int a) {
	int sum=0;
	int sum_digit;
	
	sum = a;
	for( ; sum<=10000; ) {
		sum_digit=0;
		for(l=0; pow(10, l)<=sum; l++)
			sum_digit += sum/(int)pow(10, l)%10;
		
		sum = sum + sum_digit;
		
		selfnum[sum]++;
	}
}

int main (void) {
	int selfnum[100036]={0,};
	
	for(i=1; i<=10000; i++)
		selfnum_count(selfnum, i);
		
	for(i=1; i<=10000; i++)
		if(selfnum[i]==0)
			printf("%d\n", i);
	
	return 0;
}