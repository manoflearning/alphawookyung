#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vl vector<ll>
#define vd vector<double>
#define vc vector<char>
#define vs vector<string>
#define vpi vector<pii>
#define vpl vector<pll>
#define vvi vector<vector<int>>
#define vvl vector<vector<ll>>
#define vvd vector<vector<double>>
#define vvc vector<vector<char>>
#define vvs vector<vector<string>>
#define vvpi vector<vector<pii>>
#define vvpl vector<vector<pll>>

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		int R, C;
		cin >> R >> C;

		vvi arr;
		arr.resize(R + 5, vi(C + 5));

		for (int y = 1; y <= R; y++) {
			for (int x = 1; x <= C; x++) {
				cin >> arr[y][x];
			}
		}

		int mDist = R + C - 2;

		vi cnt0(mDist + 5), cnt1(mDist + 5);

		for (int y = 1; y <= R; y++) {
			for (int x = 1; x <= C; x++) {
				if (arr[y][x] == 0) cnt0[y + x - 2]++;
				else cnt1[y + x - 2]++;
			}
		}

		int ans = 0;
		for (int i = 0; i <= mDist / 2; i++) {
			if (mDist % 2 == 0 && i == mDist / 2) continue;

			int c0 = cnt0[i] + cnt0[mDist - i], c1 = cnt1[i] + cnt1[mDist - i];

			if (c0 <= c1) ans += c0;
			else ans += c1;
		}

		cout << ans << '\n';
	}

	return 0;
}