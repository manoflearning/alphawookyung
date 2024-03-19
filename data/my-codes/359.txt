#include <iostream>

int year_count(int e, int s, int m);

int main(void) {
	int e, s, m;
	
	std::cin>>e>>s>>m;
	
	std::cout<<year_count(e, s, m);
	
	return 0;
}

int year_count(int e, int s, int m) {
	int year=1;
	
	if(e==15) e=0;
	if(s==28) s=0;
	if(m==19) m=0;
	
	while(1) {
		if(e==year%15 && s==year%28 && m==year%19) break;
		
		year++;
	}
	
	return year;
}
