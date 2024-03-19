#include <iostream>
using namespace std;
#define ll long long

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	ll n, x; 
	cin >> n;

	x = n - 1;

	ll ans = 0;

	for (ll i = n; i > 0; i = x / (x / i + 1))
		ans += (x / i + 1) * (i - (x / (x / i + 1)));

	cout << ans;

	return 0;
}