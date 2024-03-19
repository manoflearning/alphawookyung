#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int n0, n1, n2;
		cin >> n0 >> n1 >> n2;
		
		string ans;
		if (n0 > 0) {
			for (int i = 0; i < n0 + 1; i++) ans.push_back('0');
		}
		if (n2 > 0) {
			for (int i = 0; i < n2 + 1; i++) ans.push_back('1');
		}

		if (n0 > 0 && n2 > 0) n1--;
		if (ans.empty() && n1 > 0) ans.push_back('1');
		for (int i = 0; i < n1; i++) {
			if (ans.back() == '0') ans.push_back('1');
			else ans.push_back('0');
		}

		cout << ans << '\n';
	}

	return 0;
}