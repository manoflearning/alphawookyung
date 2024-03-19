#include <stdio.h>
#include <string.h>

struct D {
	char d[101];
};
int count (struct D D[], int v) {
	int i, l;
	int m1, m2;
	int count=0;
	
	for(i=0; i<v; i++) {
		for(m1=0; m1<strlen(D[i].d); m1++) {
			m2=m1+1;
			while(1) {
				if(D[i].d[m1] == D[i].d[m2])
					m2++;
				else break;
			}
			for(l=m2; l<strlen(D[i].d); l++) {
				if(D[i].d[m1]==D[i].d[l]) break;
			}
			if(l!=strlen(D[i].d)) break;
		}
		if(m1==strlen(D[i].d)) count++;
	}
	
	return count;
}
int main(void) {
	int v, i;
	struct D D[100];
	
	scanf("%d", &v);
	getchar();
	
	for(i=0; i<v; i++) 
		gets(D[i].d);
	
	printf("%d", count(D, v));
	
	return 0;
}