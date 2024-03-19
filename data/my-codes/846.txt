#include <iostream>

int k_erase_number(int n, int k);

int main(void) {
	int n, k;
	
	std::cin>>n>>k;
	
	std::cout<<k_erase_number(n, k);
	
	return 0;
}

int k_erase_number(int n, int k) {
	int count=0;
	int arr[n+1]={0};
	int ans;
	
	for(int i=2; i<=n; i++)
		if(arr[i]==0)
			for(int l=i; l<=n; l+=i)
				if(arr[l]==0) {
					count++;
					arr[l]=1;
					
					if(count==k) {
						i=n+1;
						ans = l;
						break;
					}
				}
	
	return ans;
}