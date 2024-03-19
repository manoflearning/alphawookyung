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

struct xy {
	int x, y;
};

xy p[3];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	for (int i = 0; i < 3; i++) cin >> p[i].x >> p[i].y;

	bool YES = true;
	if (p[0].x == p[1].x) {
		if (p[1].x != p[2].x) YES = false;
	}
	else if (p[0].y == p[1].y) {
		if (p[1].y != p[2].y) YES = false;
	}
	else {
		if (p[1].x == p[2].x || p[1].y == p[2].y) YES = false;
		else {
			int xx = p[1].x - p[0].x, yy = p[1].y - p[0].y;
			int xxx = p[2].x - p[1].x, yyy = p[2].y - p[1].y;
			YES = 1.0 * xx / yy == 1.0 * xxx / yyy;
		}
	}
	
	if (YES) cout << "WHERE IS MY CHICKEN?";
	else cout << "WINNER WINNER CHICKEN DINNER!";

	return 0;
}