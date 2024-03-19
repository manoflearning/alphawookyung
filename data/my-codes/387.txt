#include <iostream>
#include <cstring>

void print1(char word[]);
void print2(char word[]);

int main()
{
	char color[3][7];
	std::cin>>color[0]>>color[1]>>color[2];
	
	if(strcmp(color[0], "black") == 0)
	{
		if(strcmp(color[1], "black") == 0)
		{
			print1(color[1]);	
		}
		else
		{
			print1(color[1]);
			print2(color[2]);
		}
	}
	else
	{
		print1(color[0]);
		print1(color[1]);
		print2(color[2]);	
	}
	
	return 0;
}

void print1(char word[])
{
	if(strcmp(word, "black") == 0)
		std::cout<<'0';
	else if(strcmp(word, "brown") == 0)
		std::cout<<'1';
	else if(strcmp(word, "red") == 0)
		std::cout<<'2';
	else if(strcmp(word, "orange") == 0)
		std::cout<<'3';
	else if(strcmp(word, "yellow") == 0)
		std::cout<<'4';
	else if(strcmp(word, "green") == 0)
		std::cout<<'5';
	else if(strcmp(word, "blue") == 0)
		std::cout<<'6';
	else if(strcmp(word, "violet") == 0)
		std::cout<<'7';
	else if(strcmp(word, "grey") == 0)
		std::cout<<'8';
	else if(strcmp(word, "white") == 0)
		std::cout<<'9';
}
void print2(char word[])
{
	if(strcmp(word, "brown") == 0)
		std::cout<<'0';
	else if(strcmp(word, "red") == 0)
		std::cout<<"00";
	else if(strcmp(word, "orange") == 0)
		std::cout<<"000";
	else if(strcmp(word, "yellow") == 0)
		std::cout<<"0000";
	else if(strcmp(word, "green") == 0)
		std::cout<<"00000";
	else if(strcmp(word, "blue") == 0)
		std::cout<<"000000";
	else if(strcmp(word, "violet") == 0)
		std::cout<<"0000000";
	else if(strcmp(word, "grey") == 0)
		std::cout<<"00000000";
	else if(strcmp(word, "white") == 0)
		std::cout<<"000000000";
}
