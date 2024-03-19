#include <iostream>

int count_area(int rec[][4]);

int main(void) {
	int rec[4][4];
	
	for(int i=0; i<4; i++)
		for(int l=0; l<4; l++)
			std::cin>>rec[i][l];
	
	std::cout<<count_area(rec);
	
	return 0;
}

int count_area(int rec[][4]) {
	int area[100][100]={0};
	int count=0;
	
	for(int i=0; i<4; i++) {
		for(int l=rec[i][0]; l<rec[i][2]; l++)
			for(int j=rec[i][1]; j<rec[i][3]; j++)
				area[l][j] = 1;
	}
	
	for(int i=1; i<100; i++)
		for(int l=1; l<100; l++)
			if(area[i][l]==1) count++;
	
	return count;
}