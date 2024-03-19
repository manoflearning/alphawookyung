#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int N, oddCnt;
ll sum;
vector<ll> a;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;
	a.resize(N);
	for (ll& i : a) {
		cin >> i;
		sum += i;
		oddCnt += i % 2;
	}

	if (sum % 3) {
		cout << "NO";
		return 0;
	}
	if (sum / 3 < oddCnt) {
		cout << "NO";
		return 0;
	}
	
	cout << "YES";

	return 0;
}