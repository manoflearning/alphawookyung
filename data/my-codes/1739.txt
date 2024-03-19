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

const int MAX = 1e7;

bool isPrime[MAX + 1];
vector<int> prime(1, 2);

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

int m, n[11];
map<int, int> mp;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	getPrime();

	int tc; cin >> tc;

	for (int tt = 1; tt <= tc; tt++) {
		cin >> m;

		for (int i = 1; i <= m; i++) cin >> n[i];

		for (int i = 1; i <= m; i++) {
			set<int> s;

			ll sum = 0;
			for (int j = 0; j < n[i]; j++) sum += prime[j];

			if (sum <= MAX && isPrime[sum]) s.insert(sum);

			for (int j = n[i]; j < prime.size(); j++) {
				sum += prime[j] - prime[j - n[i]];

				if (sum <= MAX && isPrime[sum]) s.insert(sum);
			}

			for (int j : s) {
				if (mp.find(j) == mp.end()) mp.insert({ j, 1 });
				else mp.find(j)->second++;
			}
		}

		cout << "Scenario " << tt << ":\n";
		for (auto& i : mp) {
			if (i.second == m) {
				cout << i.first << "\n\n";
				break;
			}
		}

		mp.clear();
	}

	return 0;
}