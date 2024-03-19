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
#define vs vector<string>
#define vvs vector<vector<string>>
 
const int INF = 1e9 + 7;
const int MOD = 998244353;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
 
ll n, ans, p10[20];
 
int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
 
	p10[0] = 1;
	for (int i = 1; i < 20; i++)
		p10[i] = 10 * p10[i - 1];
 
	cin >> n;
 
	if (n & 1) {
		ans = ((n + 1) / 2 % MOD) * (n % MOD);
	}
	else {
		ans = (n / 2 % MOD) * ((n + 1) % MOD);
	}
 
	for (int i = 0; p10[i] <= n; i++) {
		ll a = (p10[i] - 1) % MOD;
		if (n >= p10[i + 1]) ans -= ((p10[i + 1] - p10[i]) % MOD) * (a % MOD);
		else ans -= ((n - a) % MOD) * (a % MOD);
		
		ans %= MOD;
		ans = (ans + MOD) % MOD;
	}
 
	cout << ans;
 
	return 0;
}
