#include <stdio.h>

void scan(int score[]);
void print(int score[]);

int main(void) {
	int score[151]={0};
	
	scan(score);
	
	print(score);
	
	return 0;
}

void scan(int score[]) {
	int s;
	int i, l;
	
	for(i=1; i<=8; i++) {
		scanf("%d", &s);
		
		score[s]=i;
	}
}

void print(int score[]) {
	int count=0;
	int sum=0, arr[8]={0};
	int i;
	
	for(i=0; i<151; i++) {
		if(score[i]) {
			if(count<3) count++;
			else {
				sum+=i;
				arr[score[i]-1]=score[i];
				count++;
			}
		}
	}
	
	printf("%d\n", sum);
	for(i=0; i<8; i++)
		if(arr[i])
			printf("%d ", arr[i]);
}