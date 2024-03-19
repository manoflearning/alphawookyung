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

ll a, b, c;

void input() {
    cin >> a >> b >> c;
}

ll f(ll b) {
    if (b == 1) return a % c;
    
    ll ret = f(b / 2);
    
    if (b & 1) return (ret * ret % c) * a % c;
    else return ret * ret % c;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	cout << f(b);
	
	return 0;
}
