#include <iostream>

int how_many_partial_sequence(int arr[], int n, int s, bool flag);

int main() {
	int n, s;
	std::cin>>n>>s;
	
	int arr[n+1];
	for(int i=1; i<=n; i++)
		std::cin>>arr[i];
		
	std::cout<<how_many_partial_sequence(arr, n, s, 0);
	
	return 0;
}

int how_many_partial_sequence(int arr[], int n, int s, bool flag) {
	if(n==0)
		if(s==0 && flag==1) return 1;
		else return 0;
		
	int count=0;
	
	count += how_many_partial_sequence(arr, n-1, s, flag);
	count += how_many_partial_sequence(arr, n-1, s-arr[n], 1);
	
	return count;
}
