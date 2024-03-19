#include <iostream>
#include <vector>
#include <algorithm>
#define ll long long
using namespace std;

string s;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T;
	cin >> T;

	while (T--) {
		int a, b, p;
		cin >> a >> b >> p;

		cin >> s;

		if (p < a && p < b) {
			cout << s.size() << '\n';
			continue;
		}

		for (int i = s.size() - 2; i >= 0; i--) {
			if (s[i] != s[i + 1] || i == s.size() - 2) {
				if (s[i] == 'A') {
					if (p >= a) p -= a;
					else {
						cout << i + 2 << '\n';
						i = -1;
					}
				}
				else {
					if (p >= b) p -= b;
					else {
						cout << i + 2 << '\n';
						i = -1;
					}
				}
			}

			if (i == 0) cout << 1 << '\n';
		}
	}

	return 0;
}