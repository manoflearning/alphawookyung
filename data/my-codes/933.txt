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

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N, M; 
	cin >> N >> M;

	vector<int> A(N);
	for (int& i : A) cin >> i;

	int sum = 0, l = 0, cnt = 0;
	for (int r = 1; r <= N; r++) {
		sum += A[r - 1];
		while (l < r && sum > M) sum -= A[l++];

		if (sum == M) cnt++;
	}

	cout << cnt;

	return 0;
}