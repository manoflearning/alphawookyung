#include <iostream>
#include <cstring>

int erase_to_be_anagram(char word1[], char word2[]);

int main(void) {
	char word1[1001], word2[1001];
	
	std::cin>>word1>>word2;
	
	std::cout<<erase_to_be_anagram(word1, word2);
	
	return 0;
}

int erase_to_be_anagram(char word1[], char word2[]) {
	int str1, str2;
	int count=0;
	
	str1 = strlen(word1);
	str2 = strlen(word2);
	
	for(int i=0; i<str1; i++)
		for(int l=0; l<str2; l++)
			if(word1[i] == word2[l]) {
				word1[i] = '\0';
				word2[l] = '\0';
			}
	
	for(int i=0; i<str1; i++)
		if(word1[i] != '\0') count++;
	for(int i=0; i<str2; i++)
		if(word2[i] != '\0') count++;
	
	return count;
}
