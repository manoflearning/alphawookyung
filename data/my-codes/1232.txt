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
const int MAX = 1e5, MAXD = 30;

int c[MAX + 5];
int dp[MAX + 5][MAXD + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int N, M, K;
	cin >> N >> M >> K;

	for (int i = 1; i <= N; i++) dp[i][0] = i;

	while (M--) {
		int L, R; cin >> L >> R;
		
		int mid = (L + R) >> 1;
		for (int i = L; i <= mid; i++)
			swap(dp[i][0], dp[R - (i - L)][0]);
	}
	
	for (int i = 1; i <= MAXD; i++) {
		for (int j = 1; j <= N; j++) {
			dp[j][i] = dp[dp[j][i - 1]][i - 1];
		}
	}

	for (int i = 1; i <= N; i++) {
		int x = i;
		for (int j = 0; j <= MAXD; j++)
			if (K & (1 << j)) x = dp[x][j];
		cout << x << '\n';
	}

	return 0;
}