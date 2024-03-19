#include <iostream>
#include <algorithm>
using namespace std;
#define ll long long

const ll bMAX = 1e18;

struct num {
	ll a, b;
	bool visited;

	num operator+(num rhs) {
		num ret = { a + rhs.a, b + rhs.b, true };
		if (ret.b >= bMAX) {
			ret.a += 1;
			ret.b -= bMAX;
		}
		return ret;
	}
	void print() {
		if (a == 0) cout << b;
		else {
			string sb;
			for (ll i = bMAX / 10; i >= 1; i /= 10) {
				int x = b % (i * 10) / i;
				sb.push_back(x + '0');
			}
			cout << a << sb;
		}
	}
};

num dp[105][105];

num C(int n, int m);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int n, m;
	cin >> n >> m;

	C(n, m).print();
	cout << '\n';

	return 0;
}

num C(int n, int m) {
	if (n < m) return { 0, 0, true };

	num& ret = dp[n][m];
	if (ret.visited) return ret;
	if (n == m || m == 0) return ret = { 0, 1, true };
	
	ret.visited = true;
	return ret = C(n - 1, m - 1) + C(n - 1, m);
}