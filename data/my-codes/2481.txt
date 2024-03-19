#include <bits/stdc++.h>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int a, b, c, d;
		cin >> a >> b >> c >> d;

		int x, y, x1, y1, x2, y2;
		cin >> x >> y >> x1 >> y1 >> x2 >> y2;

		int e = b - a, f = d - c;

		bool YES = true;

		if (!(x1 <= x + e && x + e <= x2 && y1 <= y + f && y + f <= y2))
			YES = false;

		if (e == 0 && (a != 0 || b != 0) && x1 == x && x == x2) YES = false;
		if (f == 0 && (c != 0 || d != 0) && y1 == y && y == y2) YES = false;

		if (YES) cout << "YES\n";
		else cout << "NO\n";
	}

	return 0;
}