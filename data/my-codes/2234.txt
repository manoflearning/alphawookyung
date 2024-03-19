#include <iostream>
#include <vector>
#include <algorithm>
#define ll long long
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T;
	cin >> T;

	while (T--) {
		int a, b, x, y;
		cin >> a >> b >> x >> y;

		int X = max(a - x - 1, x);
		int Y = max(b - y - 1, y);

		cout << max(a * Y, b * X) << '\n';
	}

	return 0;
}