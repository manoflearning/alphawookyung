#include <stdio.h>
#include <string.h>

int i, l, j;

int main (void) {
	char word[1000001];
	int count[26];
	int max1, max2, mp;
	int str;
	
	scanf("%s", word);
	
	str=strlen(word);
	for(i=0; i<str; i++) {
		for(l='A'; l<='Z'; l++) {
			if(l==word[i] || l+32==word[i])
				count[l-65]++;
		}
	}
	
	max1=count[0];
	mp=0;
	for(i=1; i<26; i++) {
		if(count[i]>=max1) {
			max2=max1;
			max1=count[i];
			mp=i;
		}
	}
	
	if(max1==max2)
		printf("?");
	else
		printf("%c", mp+65);
	
	return 0;
}
