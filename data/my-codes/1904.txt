#include <iostream>
#define ll long long
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	while (true) {
		ll a, b, c;
		cin >> a >> b >> c;

		if (a == 0 && b == 0 && c == 0) break;

		if ((c - a) % b == 0 && (c - a) / b >= 0) cout << (c - a) / b + 1 << '\n';
		else cout << "X\n";
	}

	return 0;
}