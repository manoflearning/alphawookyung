#include <stdio.h>
#include <string.h>

int wear_count();

int main(void) {
	int tc;
	int i;
	
	scanf("%d", &tc);
	
	for(i=0; i<tc; i++) {
		printf("%d\n", wear_count());
	}
	
	return 0;
}

int wear_count() {
	int n;
	char f_name[21], f_kind[21];
	char kind[30][21]={0};
	int count[30]={0};
	int result=1;
	int i, l;
	
	scanf("%d", &n);
	
	for(i=0; i<n; i++) {
		scanf("%s %s", f_name, f_kind);
		for(l=0; l<30; l++) {
			if(strcmp(f_kind, kind[l])==0) {
				strcpy(kind[l], f_kind);
				count[l]++;
				break;
			}
			else if(kind[l][0]==0) {
				strcpy(kind[l], f_kind);
				count[l]++;
				break;
			}
		}
	}
	
	for(i=0; count[i]!=0; i++)		
		result*=count[i]+1;
	
	result--;
	
	return result;
}