#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
	int tc; cin >> tc;
	while (tc--) {
	    int n; cin >> n;
	    
	    int p, q;
	    cin >> p >> q;
	    
	    vector<ll> a(n);
	    for (auto& i : a) cin >> i;
	    
	    sort(a.begin(), a.end());
	    
	    ll ans = a[n - 1] - a[0];
	    
	    priority_queue<ll> pq;
	    for (int i = 1; i < n - 1; i++) pq.push(abs(a[i]));
	    
	    int cnt = p + q;
	    while (cnt-- && !pq.empty()) {
	        ans += pq.top();
	        pq.pop();
	    }
	    
	    cout << ans << '\n';
	}
	return 0;
}
