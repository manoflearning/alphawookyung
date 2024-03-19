#include <iostream>
#define ll long long
using namespace std;

int main() {
	cin.tie(NULL);
	cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int t;
	cin >> t;

	for (int test = 0; test < t; test++) {
		ll n, m;
		cin >> n >> m;

		ll all = n * (n + 1) / 2;
		ll c0 = n - m;
		ll div = c0 / (m + 1);

		ll cnt1 = c0 % (m + 1), cnt2 = (m + 1) - cnt1;
		all -= cnt1 * (div + 1) * (div + 2) / 2;
		all -= cnt2 * div * (div + 1) / 2;
		
		cout << all << '\n';
	}

	return 0;
}