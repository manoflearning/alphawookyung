#include <stdio.h>

struct player_info {
	int atk;
	int life;
};

int main(void) {
	struct player_info pi[2];
	
	scanf("%d %d", &pi[0].atk, &pi[0].life);
	scanf("%d %d", &pi[1].atk, &pi[1].life);
	
	while(1) {
		pi[0].life -= pi[1].atk;
		pi[1].life -= pi[0].atk;
		
		if(pi[0].life<=0 || pi[1].life<=0) break;
	}
	
	if(pi[0].life > 0) printf("PLAYER A");
	else if(pi[1].life > 0) printf("PLAYER B");
	else printf("DRAW");
	
	return 0;
}