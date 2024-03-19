#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>

using namespace std;

int n;
vector<int> number;
int ad, mi, mu, di;

int max_find(int flag, int cad, int cmi, int cmu, int cdi);

int min_find(int flag, int cad, int cmi, int cmu, int cdi);

int main()
{
	scanf("%d", &n);
	
	number.resize(n);
	for(int i=0; i<n; i++)
		scanf("%d", &number[i]);
	
	scanf("%d %d %d %d", &ad, &mi, &mu, &di);
	
	printf("%d\n%d", max_find(n-1, ad, mi, mu, di), min_find(n-1, ad, mi, mu, di));
	
	return 0;
}

int max_find(int flag, int cad, int cmi, int cmu, int cdi)
{
	if(flag == 0) return number[flag];
	
	int value = INT_MIN;
	
	if(cad > 0)
		value = max(value, max_find(flag-1, cad-1, cmi, cmu, cdi) + number[flag]);
	if(cmi > 0)
		value = max(value, max_find(flag-1, cad, cmi-1, cmu, cdi) - number[flag]);
	if(cmu > 0)
		value = max(value, max_find(flag-1, cad, cmi, cmu-1, cdi) * number[flag]);
	if(cdi > 0)
		value = max(value, max_find(flag-1, cad, cmi, cmu, cdi-1) / number[flag]);
	
	return value;
}

int min_find(int flag, int cad, int cmi, int cmu, int cdi)
{
	if(flag == 0) return number[flag];
	
	int value = INT_MAX;
	
	if(cad > 0)
		value = min(value, min_find(flag-1, cad-1, cmi, cmu, cdi) + number[flag]);
	if(cmi > 0)
		value = min(value, min_find(flag-1, cad, cmi-1, cmu, cdi) - number[flag]);
	if(cmu > 0)
		value = min(value, min_find(flag-1, cad, cmi, cmu-1, cdi) * number[flag]);
	if(cdi > 0)
		value = min(value, min_find(flag-1, cad, cmi, cmu, cdi-1) / number[flag]);
	
	return value;
}