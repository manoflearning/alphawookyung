#include <stdio.h>
#include <string.h>

int count (char sentence[]) {
	int i, l;
	int count=0;

	for(i=0; sentence[i]!=NULL; i++)
		if((('A'<=sentence[i] && sentence[i] <='Z') || ('a'<=sentence[i] && sentence[i]<='z')) && (sentence[i+1]==' ' || sentence[i+1]==NULL))
			count++;
	
	return count;
}
int main (void) {
	char sentence[1000002]={0,};
	
	gets(sentence);
	
	printf("%d", count(sentence));
		
	return 0;
}
