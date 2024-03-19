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
		string s;
		int n;
		cin >> s >> n;
		
		cout << s << ' ';
		if (n >= 97) cout << "A+\n";
		else if (n >= 90) cout << "A\n";
		else if (n >= 87) cout << "B+\n";
		else if (n >= 80) cout << "B\n";
		else if (n >= 77) cout << "C+\n";
		else if (n >= 70) cout << "C\n";
		else if (n >= 67) cout << "D+\n";
		else if (n >= 60) cout << "D\n";
		else if (n >= 0) cout << "F\n";
	}

	return 0;
}