#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vl vector<ll>
#define vvl vector<vector<ll>>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int n, a[100005], b[100005];

void input() {
    cin >> n;
    for (int i = 0; i < n; i++) {
        int x; cin >> x;
        a[x]++;
    }
}

ll f() {
    ll ret = 0, prv = 0;
    for (ll i = 100000; i >= 2; i--) {
        if (a[i] && ((a[i] + b[i]) & 1)) a[i]--, b[i - 1]++;
        
        if (a[i]) {
            ll c = (a[i] + b[i]) / 2;
            
            if (prv) {
                ret += prv * i;
                prv = 0, c--;
            }
            
            ret += i * i * (c / 2);
            if (c & 1) prv = i;
        }
    }
    return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	cout << f();
	
	return 0;
}
