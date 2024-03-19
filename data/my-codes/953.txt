#include <stdio.h>

int main(){
	int j,num[9],max;
	
	for(int i=0;i<9;i++){
		scanf("%d",&num[i]);
	}
	
	max=num[0];
	j=1;
	for(int i=1;i<9;i++){
		if(num[i]>max){
			max=num[i];
			j=i+1;
		}
	}
	
	printf("%d\n%d\n",max,j);
	
	return 0;
}
