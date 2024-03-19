#include <iostream>

struct baseball {
	char bb[4];
	int s;
	int b;
};

int how_many_ans(struct baseball b[], int N);
int ball_check(struct baseball b, int i, int l, int j);
int strike_check(struct baseball b, int i, int l, int j);

int main() {
	int N;
	
	std::cin>>N;
	
	struct baseball b[N];
	for(int i=0; i<N; i++)
		std::cin>>b[i].bb>>b[i].s>>b[i].b;
	
	std::cout<<how_many_ans(b, N);
	
	return 0;
}

int how_many_ans(struct baseball b[], int N) {
	int count=0;
	
	for(int i=0; i<N; i++)
		if(b[i].s==3) return 1;
	
	for(int i=49; i<=57; i++)
		for(int l=49; l<=57; l++)
			for(int j=49; j<=57; j++) {
				int k;
				for(k=0; k<N; k++) {
					if(b[k].s==3) {
						if(strike_check(b[k], i, l, j)!=2) break;}
					else if(b[k].s==2) {
						if(strike_check(b[k], i, l, j)!=2) break;}
					else if(b[k].s==1) {
						if(strike_check(b[k], i, l, j)!=1) break;
						
						if(b[k].b==0) {
							if(ball_check(b[k], i, l, j)!=0) break;}
						else if(b[k].b==1) {
							if(ball_check(b[k], i, l, j)!=1) break;}
						else if(b[k].b==2) {
							if(ball_check(b[k], i, l, j)!=2) break;}
					}
					else if(b[k].s==0) {
						if(strike_check(b[k], i, l, j)!=0) break;
						
						if(b[k].b==0) {
							if(ball_check(b[k], i, l, j)!=0) break;}
						else if(b[k].b==1) {
							if(ball_check(b[k], i, l, j)!=1) break;}
						else if(b[k].b==2) {
							if(ball_check(b[k], i, l, j)!=2) break;}
						else if(b[k].b==3) {
							if(ball_check(b[k], i, l, j)!=3) break;}
					}
				}
				
				if(k==N)
					if(i!=l && l!=j && i!=j)
						count++;
			}
				
	return count;
}

int ball_check(struct baseball b, int i, int l, int j) {
	int c=0;
	
	for(int m=1; m<=2; m++)		
		if(b.bb[m] == i) c++;
	for(int m=0; m<=2; m+=2)
		if(b.bb[m] == l) c++;
	for(int m=0; m<=1; m++)
		if(b.bb[m] == j) c++;
							
	return c;
}
int strike_check(struct baseball b, int i, int l, int j) {
	int c=0;
	
	if(b.bb[0] == i) c++;
	if(b.bb[1] == l) c++;
	if(b.bb[2] == j) c++;
	
	return c;
}
