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
const int MAX = 3e6;

bool isPrime[MAX + 10];
int dp[MAX + 1];

void getPrime() {
	fill(isPrime, isPrime + MAX + 1, true);

	for (ll i = 4; i <= MAX; i += 2)
		isPrime[i] = false;
	for (ll i = 3; i <= MAX; i++) {
		for (ll j = i * i; j <= MAX; j += i * 2)
			isPrime[j] = false;
	}
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	getPrime();

	for (int i = 1; i <= 1e6; i++) {
		dp[i] = dp[i - 1];
		if (isPrime[3 * i + 7]) dp[i]++;
	}

	int t; cin >> t;
	
	while (t--) {
		int n; cin >> n;

		cout << dp[n] << '\n';
	}

	return 0;
}
