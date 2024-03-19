#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define sz(x) (int)(x).size()
#define all(x) (x).begin(), (x).end()

int b, n;
ll a[4040], psum[4040], psum2[4040];
bool vi[4040][33];
double dp[4040][33], pre[4040][4040];

void init() {
	psum[0] = a[0];
	for (int i = 1; i < n; i++)
		psum[i] = a[i] + psum[i - 1];

	psum2[0] = a[0] * a[0];
	for (int i = 1; i < n; i++)
		psum2[i] = a[i] * a[i] + psum2[i - 1];
}

double f(int st, int cnt) {
	double& ret = dp[st][cnt];
	if (vi[st][cnt]) return ret;
	if (cnt == 1) return ret = pre[st][n - 1];

	vi[st][cnt] = 1;
	ret = pre[st][n - 1];

	for (int i = st; i < n; i++) {
		ret = min(ret, pre[st][i] + f(i + 1, cnt - 1));
	}

	return ret;
}

ll q(int l, int r) {
	return psum[r] - (l == 0 ? 0 : psum[l - 1]);
}

ll q2(int l, int r) {
	return psum2[r] - (l == 0 ? 0 : psum2[l - 1]);
}

double aver(int l, int r) {
	double res = q(l, r);
	return res / (r - l + 1);
}

void preprocess() {
	for (int l = 0; l < n; l++) {
		for (int r = l; r < n; r++) {
			double av = aver(l, r);
			pre[l][r] = q2(l, r) - 2 * av * q(l, r) + (r - l + 1) * av * av;
		}
	}
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	cout << fixed;
	cout.precision(8);

	cin >> b >> n;
	for (int i = 0; i < n; i++) cin >> a[i];

	init();

	preprocess();

	cout << f(0, b);

	return 0;
}