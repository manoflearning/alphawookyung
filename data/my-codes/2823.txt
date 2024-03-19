#include <iostream>
using namespace std;
#define ll long long

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	ll N; cin >> N;

	ll ans = 0;
	for (ll i = 1; i <= N; i++)
		ans += (i + N / i * i) * (N / i) / 2;

	cout << ans << '\n';

	return 0;
}