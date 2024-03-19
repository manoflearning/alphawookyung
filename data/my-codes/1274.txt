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

vector<int> a;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	for (int i = 1; i * (i + 1) / 2 <= 1000; i++) {
	    a.push_back(i * (i + 1) / 2);
	}
	
	int t; cin >> t;
	
	while (t--) {
	    int n; cin >> n;
	    
	    int ans = 0;
	    
	    for (int i = 0; i < a.size(); i++) {
	        for (int j = i; j < a.size(); j++) {
	            for (int k = j; k < a.size(); k++) {
	                if (a[i] + a[j] + a[k] == n) {
	                    ans = 1;
	                    i = j = k = a.size() - 1;
	                }
	            }
	        }
	    }
	    
	    cout << ans << '\n';
	}
	
	return 0;
}
