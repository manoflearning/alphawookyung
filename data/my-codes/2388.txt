#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const long double PI = 3.141592653589793238462643383279502884197;

const int MAX = 1e5;

bool isPrime[MAX + 1];
vector<int> prime = { 1, 2 };

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

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	getPrime();

	int tc; cin >> tc;

	while (tc--) {
		int n; cin >> n;

		vector<int> arr(n + 1);

		bool all0 = true, all1 = true;
		bool exist1after0 = false, exist1btw0 = false;

		for (int i = 1; i <= n; i++) cin >> arr[i];

		for (int i = 1; i <= n; i++) {
			if (arr[i] == i) all0 = false;
			else all1 = false;

			if (arr[i] == i && !all1) exist1after0 = true;
			if (exist1after0 && arr[i] != i) exist1btw0 = true;
		}

		if (all0) cout << 1 << '\n';
		else if (all1) cout << 0 << '\n';
		else {
			if (exist1btw0) cout << 2 << '\n';
			else cout << 1 << '\n';
		}
	}

	return 0;
}