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

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;
	while (tc--) {
		int n, m;
		cin >> n >> m;

		vector<string> a(n);
		for (int i = 0; i < n; i++) {
			cin >> a[i];
		}

		int res = 0;
		for (int i = 0; i < n - 1; i++)
			if (a[i].back() != 'D') res++;
		for(int i=0; i<m-1; i++)
            if(a[n-1][i] != 'R') res++;
		cout << res << '\n';
	}

	return 0;
}