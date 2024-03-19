#include <stdio.h>
#include <string.h>

int CroatianAlphabet_Count (char CA[]) {
	int i;
	int count=0;
	
	for(i=0; i<strlen(CA); ) {
		if(CA[i]=='c') {
			if(CA[i+1]=='=') {
				count++;
				i=i+2;
			}
			else if(CA[i+1]=='-') {
				count++;
				i=i+2;
			}
			else {
				count++;
				i++;
			}
		}
		else if(CA[i]=='d') {
			if(CA[i+1]=='z' && CA[i+2]=='=') {
				count++;
				i=i+3;
			}
			else if(CA[i+1]=='-') {
				count++;
				i=i+2;
			}
			else {
				count++;
				i++;
			}
		}
		else if(CA[i]=='l' && CA[i+1]=='j') {
			count++;
			i=i+2;
		}
		else if(CA[i]=='n' && CA[i+1]=='j') {
			count++;
			i=i+2;
		}
		else if(CA[i]=='s' && CA[i+1]=='=') {
			count++;
			i=i+2;
		}
		else if(CA[i]=='z' && CA[i+1]=='=') {
			count++;
			i=i+2;
		}
		else {
			count++;
			i++;
		}
	}
	
	return count;
}
int main (void) {
	char CA[101];
	int count;
	
	gets(CA);
		
	count = CroatianAlphabet_Count(CA);
		
	printf("%d\n", count);
	
	return 0;
}