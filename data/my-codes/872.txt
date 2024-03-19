#include <stdio.h>

void primenumber(int M, int N) {
	int minimum, total=0;
	int flag;
	int i1, i2, l;
	
	for(i1=M; i1<=N; i1++) {
		flag=0;
		for(l=1; l<=i1; l++)
			if(i1%l==0) flag++;
		
		if(flag==2)  {
			total+=i1;
			minimum=i1;
			break;
		}
	}
	for(i2=i1+1; i2<=N; i2++) {
		flag=0;
		for(l=1; l<=i2; l++)
			if(i2%l==0) flag++;
		
		if(flag==2) total+=i2;
	}
	
	if(total==0)
		printf("-1");
	else
		printf("%d\n%d", total, minimum);
}

int main (void) {
	int M, N;
	
	scanf("%d %d", &M, &N);
	
	primenumber(M, N);
	
	return 0;
}