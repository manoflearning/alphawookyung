#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vl vector<ll>
#define vb vector<bool>
#define vd vector<double>
#define vc vector<char>
#define vs vector<string>
#define vpi vector<pii>
#define vpl vector<pll>
#define vvi vector<vector<int>>
#define vvl vector<vector<ll>>
#define vvb vector<vector<bool>>
#define vvd vector<vector<double>>
#define vvc vector<vector<char>>
#define vvs vector<vector<string>>
#define vvpi vector<vector<pii>>
#define vvpl vector<vector<pll>>

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const long double PI = 3.141592653589793238462643383279502884197;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		ll n, k;
		cin >> n >> k;

		vl arr(n);
		for (ll& i : arr) cin >> i;

		vl rem;
		for (int i = 0; i < n; i++) {
			if (arr[i] % k == 0) continue;
			rem.push_back(k - arr[i] % k);
		}

		sort(rem.begin(), rem.end());

		ll ans = 0, same = 0;
		for (int i = 0; i < rem.size(); i++) {
			if (i > 0 && rem[i - 1] == rem[i]) same++;
			else same = 0;

			ans = max(ans, 1 + rem[i] + same * k);
		}

		cout << ans << '\n';
	}

	return 0;
}