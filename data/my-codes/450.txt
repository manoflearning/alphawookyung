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

const int MAX = 1e7;

ll a, b;
bool isPrime[MAX + 1];
vector<ll> prime(1, 2);

void getPrime() {
	fill(isPrime, isPrime + MAX + 1, 1);

	for (ll i = 4; i <= MAX; i += 2)
		isPrime[i] = 0;
	for (ll i = 3; i <= MAX; i++) {
		if (isPrime[i]) prime.push_back(i);
		for (ll j = i * i; j <= MAX; j += i * 2)
			isPrime[j] = 0;
	}
}

void input() {
	cin >> a >> b;
}

int f() {
	int ret = 0;
	for (ll i : prime) {
		for (ll j = i * i; j <= b; j *= i) {
			if (a <= j) ret++;
			if (j > LLONG_MAX / i) break;
		}
	}
	return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	getPrime();
	
	input();

	cout << f();

	return 0;
}