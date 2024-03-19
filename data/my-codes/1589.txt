#include <stdio.h>

int namescore(int name_length, char name[]);

int main(void) {
	int name_length;
	char name[101];
	
	scanf("%d %s", &name_length, name);
	
	printf("%d", namescore(name_length, name));
	
	return 0;
}

int namescore(int name_length, char name[]) {
	int score=0;
	int i;
	
	for(i=0; i<name_length; i++) 
		score+=name[i]-64;
	
	return score;
}