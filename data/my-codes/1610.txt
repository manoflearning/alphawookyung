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

const int MAXN = 1e5;

int l, n;
ll a, b[MAXN + 5], c[MAXN + 5], mx[MAXN + 5];

void input() {
    int x, y;
    cin >> l >> n >> x >> y;
    a = x - y;
    
    for (int i = 0; i < n; i++) 
        cin >> b[i] >> c[i];
}

ll f() {
    ll ret = 0;
    
    for (int i = n - 1; i >= 1; i--) {
        mx[i] = max(mx[i + 1], c[i]);
        ret += a * (b[i] - b[i - 1]) * mx[i];
    }
    mx[0] = max(mx[1], c[0]);
    ret += a * b[0] * mx[0];
    
    return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	cout << f();
	
	return 0;
}
