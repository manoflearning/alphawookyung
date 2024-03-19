#include <iostream>
#define ll long long
using namespace std;

bool exist[10];

ll gcd(ll a, ll b);
ll lcm(ll a, ll b);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	ll N; cin >> N;

	for (ll i = 1; i <= N; i *= 10)
		exist[N / i % 10] = true;

	ll LCM = 1;
	for (int i = 1; i <= 9; i++)
		if (exist[i]) LCM = lcm(LCM, i);
	
	if (N % LCM == 0) {
		cout << N << '\n';
		return 0;
	}

	for (ll s = 10; ; s *= 10) {
		for (ll p = 0; p < s; p++) {
			ll val = s * N + p;

			if (val % LCM == 0) {
				cout << val << '\n';
				return 0;
			}
		}
	}

	return 0;
}

ll gcd(ll a, ll b) {
	if (b == 0) return a;
	return gcd(b, a % b);
}

ll lcm(ll a, ll b) {
	return a * b / gcd(a, b);
}