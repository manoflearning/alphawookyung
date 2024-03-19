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

	string A, B;
	cin >> A >> B;

	int ans = INF;

	for (int i = 0; i < B.size() - A.size() + 1; i++) {
		int diff = 0;
		for (int j = 0; j < A.size(); j++) {
			if (A[j] != B[j + i]) diff++;
		}
		ans = min(ans, diff);
	}

	cout << ans;

	return 0;
}