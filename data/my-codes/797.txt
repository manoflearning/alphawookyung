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

int M, a[10005];
int psum[10005];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;
	while (tc--) {
		cin >> M;
		for (int i = 1; i <= M; i++) {
			cin >> a[i];
			psum[i] = a[i] + psum[i - 1];
		}

		for (int i = 1; i <= M; i++) {
			if (psum[M] % psum[i]) continue;
			bool yes = true;
			int l = i;
			for (int j = psum[i]; j <= psum[M]; j += psum[i]) {
				while (l < M && psum[l] < j) l++;
				if (psum[l] > j) {
					yes = false;
					break;
				}
			}
			if (yes) {
				cout << psum[i] << '\n';
				break;
			}
		}
	}

	return 0;
}