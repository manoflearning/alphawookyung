#include <iostream>
#include <algorithm>
#define ll long long
using namespace std;

int main() {
	cin.tie(NULL);
	cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T;
	cin >> T;

	while (T--) {
		ll n, x, y;
		cin >> n >> x >> y;

		ll Xfront = (x - 1), Xback = max((ll)0, (n - x - 1));
		ll Yfront = (y - 1), Yback = max((ll)0, (n - y - 1));

		ll ans1 = (x - 1) + (y - 1) + 1;
		
		if (Yback >= Xfront && Xback >= Yfront) ans1 = 1;
		else if (Yback >= Xfront || Xback >= Yfront) {
			if (Yback >= Xfront) {
				ans1 -= Xfront;
				ans1 -= Xback;
			}
			else {
				ans1 -= Yfront;
				ans1 -= Yback;
			}
		}
		else {
			ans1 = max(Xfront - Yback, Yfront - Xback) + 1;
		}

		ll ans2 = 1;
		Xback = (n - x);
		Yback = (n - y);
		if (Xback <= Yfront && Xfront <= Yback) {
			ans2 = n;
		}
		else if (Xback <= Yfront || Xfront <= Yback) {
			if (Xback <= Yfront) {
				ans2 += Xback;
				ans2 += Yback;
				ans2 += max(Yfront - Xback, Xfront - Yback);
			}
			else {
				ans2 += Xfront;
				ans2 += Yfront;
			}
		}
		
		cout << ans1 << ' ' << ans2 << '\n';
	}

	return 0;
}