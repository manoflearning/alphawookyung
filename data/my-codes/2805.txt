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

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int X; cin >> X;

	if (X < 600) cout << 8;
	else if (X < 800) cout << 7;
	else if (X < 1000) cout << 6;
	else if (X < 1200) cout << 5;
	else if (X < 1400) cout << 4;
	else if (X < 1600) cout << 3;
	else if (X < 1800) cout << 2;
	else if (X < 2000) cout << 1;

	return 0;
}
