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
		int n; cin >> n;

		int cnt2 = 0, cnt3 = 0;
		while (n % 2 == 0) {
			cnt2++;
			n /= 2;
		}
		while (n % 3 == 0) {
			cnt3++;
			n /= 3;
		}

		if (n == 1 && cnt2 <= cnt3) cout << cnt3 + (cnt3 - cnt2) << '\n';
		else cout << -1 << '\n';
	}

	return 0;
}