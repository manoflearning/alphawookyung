#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#define ll long long
using namespace std;

vector<int> pi, dp(1e6 + 5, -1);

void getpi(const string& P);
int f(int x);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int n;
	string S;
	cin >> n >> S;

	getpi(S);

	ll ans = 0;
	for (int i = 0; i < S.size(); i++) {
		int val = f(i);
		if (val < 1e9) ans += (i + 1) - val;
	}

	cout << ans << '\n';

	return 0;
}

void getpi(const string& P) {
	pi.resize(P.size());

	int begin = 1, matched = 0;
	while (begin + matched < P.size()) {
		if (P[begin + matched] == P[matched]) {
			matched++;
			pi[begin + matched - 1] = matched;
		}
		else {
			if (matched == 0) begin++;
			else {
				begin += matched - pi[matched - 1];
				matched = pi[matched - 1];
			}
		}
	}
}

int f(int x) {
	if (x < 0) return 1e9;
	int& ret = dp[x];
	if (ret != -1) return ret;
	if (pi[x] == 0) return ret = 1e9;
	return ret = min(pi[x], f(pi[x] - 1));
}