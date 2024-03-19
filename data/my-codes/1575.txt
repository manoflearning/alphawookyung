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

	string s; cin >> s;

	for (int i = 0; i < s.size(); i++) {
		if (s[i] == 'x') {
			if (i == 0) cout << '1';
			else if (i == 1 && s[0] == '-') cout << "-1";
			else for (int j = 0; j < i; j++) cout << s[j];
			break;
		}

		if (i == s.size() - 1) cout << '0';
	}

	return 0;
}