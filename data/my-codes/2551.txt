#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

char a[101], b[101], c[101];

int main() {
	cin.tie(NULL);
	cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int t;
	cin >> t;

	for (int test = 0; test < t; test++) {
		cin >> a >> b >> c;

		int len = strlen(a);

		bool ans = true;
		for (int i = 0; i < len; i++) {
			if (a[i] == c[i]) continue;
			if (b[i] == c[i]) continue;
			ans = false;
			break;
		}

		if (ans) cout << "YES\n";
		else cout << "NO\n";
	}

	return 0;
}