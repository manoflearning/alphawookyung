#include <stdio.h>

int main(){
	int um[8],flag=0;
	
	for(int i=0;i<8;i++)
		scanf("%d",&um[i]);
	
	for(int i=0;i<8;i++){
		if(um[i]==i+1)
			flag++;
		if(um[i]==8-i)
			flag--;
	}
	
	if(flag==8)
		printf("ascending\n");
	else if(flag==-8)
		printf("descending\n");
	else
		printf("mixed\n");
		
	return 0;
}