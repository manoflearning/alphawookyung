#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const long double PI = 3.141592653589793238462643383279502884197;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	string S;
	getline(cin, S);

	bool one = false, two = false, three = false, four = false;
	for (char c : S) {
		if (!one) {
			if (c == 'U') one = true;
		}
		else if (!two) {
			if (c == 'C') two = true;
		}
		else if (!three) {
			if (c == 'P') three = true;
		}
		else if (!four) {
			if (c == 'C') four = true;
		}
	}

	if (one && two && three && four) cout << "I love UCPC\n";
	else cout << "I hate UCPC\n";

	return 0;
}