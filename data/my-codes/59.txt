#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main() {
	int tc; cin >> tc;
	while (tc--) {
	    int n; cin >> n;
	    vector<ll> a(n);
	    for (auto& i : a) cin >> i;
	    ll sum = 0;
	    for (auto& i : a) sum += i;
	    
	    int YES = 1;
	    YES &= (sum % n ? 0 : 1);
	    for (auto& i : a) {
	        if (abs(i - sum / n) % 2) YES = 0;
	    }
	    
	    if (YES) cout << "Yes\n";
	    else cout << "No\n";
	}
	return 0;
}
