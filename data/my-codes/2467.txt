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
		int n, x, m;
		cin >> n >> x >> m;

		int mL = x, mR = x;

		while (m--) {
			int L, R;
			cin >> L >> R;

			if (R < mL || mR < L) continue;

			mL = min(mL, L);
			mR = max(mR, R);
		}

		cout << mR - mL + 1 << '\n';
	}

	return 0;
}