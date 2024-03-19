#include <stdio.h>

int primenumber_count(int N, int numbers[]) {
	int flag;
	int count=0;
	int i, l;
	
	for(i=0; i<N; i++) {
		flag=0;
		for(l=1; l<=numbers[i]; l++)
			if(numbers[i]%l==0) flag++;
			
		if(flag==2) count++;
	}
	
	return count;
}

int main (void) {
	int N, numbers[100];
	int i;
	
	scanf("%d", &N);
	
	for(i=0; i<N; i++)
		scanf("%d", &numbers[i]);
	
	printf("%d", primenumber_count(N, numbers));
	
	return 0;
}
