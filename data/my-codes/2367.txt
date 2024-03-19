#include <iostream>
#include <vector>
#include <algorithm>
#define ll long long
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	vector<ll> arr(N);
	for (ll& i : arr) cin >> i;

	ll ans = 0;
	for (int i = 0; i < N; i++) {
		ans = max(ans, arr[i]);
		for (int l = i + 1; l < N; l++) {
			ans = max(ans, arr[i] | arr[l]);
			for (int j = l + 1; j < N; j++) {
				ans = max(ans, arr[i] | arr[l] | arr[j]);
			}
		}
	}

	cout << ans << '\n';

	return 0;
}