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

const int MAX = 1e6;

int n;
ll a[105];
vi b[MAX + 1];

bool isPrime[MAX + 1];
vector<ll> prime(1, 2);

void getPrime() {
	fill(isPrime, isPrime + MAX + 1, true);

	for (ll i = 4; i <= MAX; i += 2)
		isPrime[i] = false;
	for (ll i = 3; i <= MAX; i++) {
		if (isPrime[i]) prime.push_back(i);
		for (ll j = i * i; j <= MAX; j += i * 2)
			isPrime[j] = false;
	}
}

void input() {
	cin >> n;
	for (int i = 0; i < n; i++) cin >> a[i];
}

void fac() {
	for (int i = 0; i < n; i++) {
		for (auto& p : prime) {
			int cnt = 0;
			while (a[i] % p == 0) {
				cnt++, a[i] /= p;
			}
			b[p].push_back(cnt);
		}
	}
}

void f() {
	ll ans = 1;
	int cnt = 0;

	for (int i = 0; i <= MAX; i++) {
		if (b[i].empty()) continue;

		sort(b[i].begin(), b[i].end());

		int sum = 0;
		for (auto j : b[i]) sum += j;

		int cut = sum / b[i].size();
		for (int j = 0; j < cut; j++) ans *= i;

		for (auto j : b[i]) {
			if (cut <= j) break;
			cnt += cut - j;
		}
	}

	cout << ans << ' ' << cnt;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	getPrime();

	input();

	fac();

	f();

	return 0;
}