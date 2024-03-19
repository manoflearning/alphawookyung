#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int n;
vector<ll> a, b;
ll cnt1, cnt0, ans;

void input() {
	cin >> n;
	
	for (int  i = 0; i < n; i++) {
		int x; cin >> x;
		if (x > 1) a.push_back(x);
		if (x == 1) cnt1++;
		if (x == 0) cnt0++;
		if (x < 0) b.push_back(x);
	}
}

ll f() {
	ll ret = 0;
	for (int i = 0; i < (int)a.size(); i += 2) {
		if (i != (int)a.size() - 1) ret += a[i] * a[i + 1];
		else ret += a[i];
	}
	for (int i = 0; i < (int)b.size(); i += 2) {
		if (i != (int)b.size() - 1) ret += b[i] * b[i + 1];
		else if (!cnt0) ret += b[i];
	}
	ret += cnt1;
	
	return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	sort(a.begin(), a.end(), greater<int>());
	sort(b.begin(), b.end());
	
	cout << f();
	
	return 0;
}
