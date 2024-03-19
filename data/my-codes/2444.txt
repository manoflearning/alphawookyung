#include <bits/stdc++.h>
#define ll long long
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int n; cin >> n;

		string x; cin >> x;
		
		string a, b;
		
		bool aBig = false;

		for (char c : x) {
			if (!aBig) {
				if (c == '2') {
					a.push_back('1');
					b.push_back('1');
				}
				else if (c == '1') {
					aBig = true;
					a.push_back('1');
					b.push_back('0');
				}
				else {
					a.push_back('0');
					b.push_back('0');
				}
			}
			else {
				a.push_back('0');
				b.push_back(c);
			}
		}

		cout << a << '\n' << b << '\n';
	}

	return 0;
}