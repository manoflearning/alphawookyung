#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;

bool cmp(int a, int b) {
	return a > b;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N, K;
		cin >> N >> K;

		vector<int> a(N), b(N);

		for (int& i : a) cin >> i;
		for (int& i : b) cin >> i;

		sort(a.begin(), a.end());
		sort(b.begin(), b.end(), cmp);

		for (int i = 0; i < K; i++) {
			if (a[i] >= b[i]) break;
			swap(a[i], b[i]);
		}

		int sum = 0;
		for (int i : a) sum += i;
		cout << sum << '\n';
	}

	return 0;
}