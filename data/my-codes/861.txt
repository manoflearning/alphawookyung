#include <iostream>
#include <cstdio>

int main()
{
	int n;
	scanf("%d", &n);
	int arr[n];
	for(int i=0; i<n; i++)
		scanf("%d", &arr[i]);
	
	for(int i=1; i<n; i++)
	{
		int tmp = arr[i];
		int l;
		for(l=i-1; l>=0; l--)
		{
			if(arr[l] > tmp)
				arr[l+1] = arr[l];
			else
			{
				break;
			}
		}
		
		arr[l+1] = tmp;
	}
	
	for(int i=0; i<n; i++)
		printf("%d\n", arr[i]);
	
	return 0;
}