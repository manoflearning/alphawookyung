#include <stdio.h>
#include <string.h>

int i, l, j;

void translate(char *word) {
	for(i=0; i<strlen(word); i++) {
		if(word[i]<=67) word[i]='2';
		else if(word[i]<=70) word[i]='3';
		else if(word[i]<=73) word[i]='4';
		else if(word[i]<=76) word[i]='5';
		else if(word[i]<=79) word[i]='6';
		else if(word[i]<=83) word[i]='7';
		else if(word[i]<=86) word[i]='8';
		else if(word[i]<=90) word[i]='9';
	}
}

int time_cal(char *word) {
	int sum=0;
	
	for(i=0; i<strlen(word); i++)
		sum+=word[i]-48+1;
	
	return sum;
}

int main (void) {
	char word[16];
	
	scanf("%s", word);
	
	translate(word);
	
	printf("%d", time_cal(word));
	
	return 0;
}