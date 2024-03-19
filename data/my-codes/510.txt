#include <iostream>
#include <vector>

using namespace std;

int n, m;
vector<int> arr;

int print_count();

int main()
{
	int t;
	scanf("%d", &t);
	
	for(int i=0; i<t; i++)
	{
		scanf("%d %d", &n, &m);
		
		arr.resize(n);
		
		for(int i=0; i<n; i++)
			scanf("%d", &arr[i]);
			
		printf("%d\n", print_count());
	}
	
	return 0;
}

int print_count()
{	
	int count = 0;
	while(1)
	{
		for(int i=0; i<n; i++)
		{	
			bool flag = true;
			
			for(int l=i+1; l<n; l++)
				if(arr[l] > arr[i])
				{
					flag = false;
					break;
				}
			
			if(flag == true)
				for(int l=0; l<i; l++)
					if(arr[l] > arr[i])
					{
						flag = false;
						break;
					}
				
			if(flag == true)
			{
				count++;
				if(i == m) return count;
				arr[i] = 0;
			}
		}
	}
}
