#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

set<int> s[25];
bool isNot[25][25];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int K, N;
	cin >> K >> N;

	vector<int> a(N);
	for (int i = 0; i < K; i++) {
		for (int& j : a) cin >> j;

		for (int l = 0; l < N; l++) {
			for (int r = l + 1; r < N; r++) {
				if (s[a[r]].find(a[l]) != s[a[r]].end()) 
					isNot[a[l]][a[r]] = isNot[a[r]][a[l]] = true;

				s[a[l]].insert(a[r]);
			}
		}
	}

	int cnt = 0;
	for (int l = 1; l <= N; l++) {
		for (int r = l + 1; r <= N; r++) {
			if (!isNot[l][r]) cnt++;
		}
	}

	cout << cnt;

	return 0;
}