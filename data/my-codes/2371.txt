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
		string t; cin >> t;

		bool exist0 = false, exist1 = false;
		for (char x : t) {
			if (x == '0') exist0 = true;
			if (x == '1') exist1 = true;
		}

		if (exist0 && exist1) {
			cout << t[0];
			for (int i = 1; i < t.size(); i++) {
				if (t[i - 1] == t[i]) {
					cout << (t[i - 1] == '0') ? '1' : '0';
				}
				cout << t[i];
			}
			cout << '\n';
		}
		else cout << t << '\n';
	}

	return 0;
}