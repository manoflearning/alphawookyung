#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int n, l;
vector<int> a;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	cin >> n >> l;
	
	a.resize(n);
	for (int& i : a) cin >> i;
	
	sort(a.begin(), a.end());
	
	int ans = 1;
	for (int i = 0; i < n; ) {
	    int f = i;
	    while (i < n && a[i] - a[f] < l) i++;
	    
	    if (i < n) ans++;
	}
	
	cout << ans;
	
	return 0;
}
