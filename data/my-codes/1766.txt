#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
using namespace std;
#define ll long long

ll AB[4000 * 4000 + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int n; cin >> n;

	vector<ll> A(n), B(n), C(n), D(n);
	for (int i = 0; i < n; i++) {
		cin >> A[i] >> B[i] >> C[i] >> D[i];
	}

	for (int i = 0; i < n; i++) {
		for (int l = 0; l < n; l++) {
			AB[n * i + l] = A[i] + B[l];
		}
	}

	sort(AB, AB + n * n);

	ll ans = 0;
	for (int i = 0; i < n; i++) {
		for (int l = 0; l < n; l++) {
			ans += upper_bound(AB, AB + n * n, -(C[i] + D[l])) - lower_bound(AB, AB + n * n, -(C[i] + D[l]));
		}
	}

	cout << ans;

	return 0;
}
