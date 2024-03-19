#include <iostream>
#include <algorithm>
#define ll long long
using namespace std;

int main() {
	cin.tie(NULL);
	cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int t;
	cin >> t;

	for (int test = 0; test < t; test++) {
		ll n, g, b;
		cin >> n >> g >> b;

		ll mid = (n + 1) / 2;
		ll bad = n - mid;

		if (mid % g == 0) cout << max(n, mid / g * (g + b) - b) << '\n';
		else cout << max(n, mid / g * (g + b) + mid % g) << '\n';
	}

	return 0;
}