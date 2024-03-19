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

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;
	while (tc--) {
		int n;
		string s;
		cin >> n >> s;

		set<int> need0, need1;
		vector<int> res(n);
		for (int i = 0; i < s.size(); i++) {
			char c = s[i];

			if (c == '0') {
				if (need0.empty()) {
					res[i] = need0.size() + need1.size() + 1;
					need1.insert(res[i]);
				}
				else {
					res[i] = *need0.begin();
					need1.insert(*need0.begin());
					need0.erase(need0.begin());
				}
			}
			else {
				if (need1.empty()) {
					res[i] = need0.size() + need1.size() + 1;
					need0.insert(res[i]);
				}
				else {
					res[i] = *need1.begin();
					need0.insert(*need1.begin());
					need1.erase(need1.begin());
				}
			}
		}

		cout << need0.size() + need1.size() << '\n';
		for (int& i : res) cout << i << ' ';
		cout << '\n';
	}

	return 0;
}