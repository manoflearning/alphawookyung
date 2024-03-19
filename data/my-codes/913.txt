#include <stdio.h>

int find_card(int n, int m, int card[]) {
	int i, l, j;
	int max=0;
	
	for(i=0; i<n-2; i++)
		for(l=i+1; l<n-1; l++)
			for(j=l+1; j<n; j++) {
				if(card[i]+card[l]+card[j]<=m)
					if(max<card[i]+card[l]+card[j])
						max=card[i]+card[l]+card[j];
			}
	
	return max;
}

int main (void) {
	int i;
	int n, m;
	int card[100];
		
	scanf("%d %d", &n, &m);
	
	for(i=0; i<n; i++)
		scanf("%d", &card[i]);
		
	printf("%d", find_card(n, m, card));
	
	return 0;
}