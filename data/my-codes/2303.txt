#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vl vector<ll>
#define vvi vector<vector<int>>
#define vvl vector<vector<ll>>
using namespace std;

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int tc; cin >> tc;

	while (tc--) {
		int n; cin >> n;

		set<int> s;
		for (int i = 0; i < n; i++) {
			int x; cin >> x;
			s.insert(x);
		}

		for (int k = 1; k < 1024; k++) {
			set<int> fs = s;

			bool ans = true;
			for (int x : s) {
				if (fs.find(x) == fs.end()) continue;

				if (s.find(x ^ k) == s.end()) {
					ans = false;
					break;
				}

				fs.erase(x);
				fs.erase(x ^ k);
			}

			if (ans) {
				cout << k << '\n';
				break;
			}
			if (k == 1023) {
				cout << -1 << '\n';
			}
		}
	}

	return 0;
}