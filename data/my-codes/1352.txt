#include <iostream>
using namespace std;

int main() {
	int n;
	scanf("%d", &n);
	
	int count = 0;
	
	for (int i = 0; i < 5; i++) {
		int sam;
		scanf("%d", &sam);
		
		if (sam == n) count++;
	}
	
	printf("%d", count);
	
	return 0;
}