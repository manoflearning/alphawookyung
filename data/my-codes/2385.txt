#include <iostream>
#include <algorithm>
#define ll long long
using namespace std;

ll m[1001];

int main() {
	cin.tie(NULL);
	cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int n;
	cin >> n;

	for (int i = 1; i <= n; i++)
		cin >> m[i];

	ll max = -1, flag = 0;
	for (int i = 1; i <= n; i++) {
		ll sam = m[i], check1 = m[i], check2 = m[i];
		for (int j = i - 1; j >= 1; j--) {
			check1 = min(check1, m[j]);
			sam += check1;
		}
		for (int j = i + 1; j <= n; j++) {
			check2 = min(check2, m[j]);
			sam += check2;
		}

		if (sam > max) {
			flag = i;
			max = sam;
		}
	}

	ll ans[1001], check1 = m[flag], check2 = m[flag];
	ans[flag] = m[flag];
	for (int i = flag - 1; i >= 1; i--) {
		check1 = min(check1, m[i]);
		ans[i] = check1;
	}
	for (int i = flag + 1; i <= n; i++) {
		check2 = min(check2, m[i]);
		ans[i] = check2;
	}

	for (int i = 1; i <= n; i++)
		cout << ans[i] << ' ';

	return 0;
}