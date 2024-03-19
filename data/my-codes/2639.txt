#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const long double PI = 3.141592653589793238462643383279502884197;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		int n;
		string s;
		cin >> n >> s;

		int opened = 0, used = 0;
		for (int i = 0; i < n; i++) {
			if (s[i] == '(') opened++;
			else {
				if (opened == 0) used++;
				else opened--;
			}
		}

		cout << used << '\n';
	}

	return 0;
}