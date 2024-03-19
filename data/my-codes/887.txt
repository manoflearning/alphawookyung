#include <stdio.h>

int cupholder(int n, char seat_info[]);

int main(void) {
	int n;
	char seat_info[51];
	
	scanf("%d", &n);
	scanf("%s", seat_info);
	
	printf("%d", cupholder(n, seat_info));
	
	return 0;
}

int cupholder(int n, char seat_info[]) {
	int count=2;
	int lc=0;
	int i;
	
	for(i=1; i<n; i++) {
		if(seat_info[i-1]=='S')
			count++;
		else {
			lc++;
			
			if(lc%2==0) count++;
		}
	}
	
	if(n >= count)
		return count;
	else
		return n;
}