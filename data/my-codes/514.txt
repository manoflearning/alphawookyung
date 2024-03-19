#include <iostream>
#include <cstring>

int main()
{
	int n;
	std::cin>>n;
	
	char word[n][51];
	for(int i=0; i<n; i++)
		std::cin>>word[i];
	
	for(int i=0; i<strlen(word[0]); i++)
	{
		int flag=0;
		for(int l=1; l<n; l++)
		{
			if(word[l-1][i]!=word[l][i])
			{
				flag++;
				break;
			}
		}
		if(flag==0) std::cout<<word[0][i];
		else std::cout<<'?';
	}
	
	return 0;
}
