#include <iostream>
#include <vector>
#include <algorithm>
#define ll long long
using namespace std;

struct point {
	ll x, y;
};

vector<point> p;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int N; cin >> N;

	for (int i = 0; i < N; i++) {
		int x, y;
		cin >> x >> y;
		p.push_back({ x, y });
	}

	if (N == 1) {
		cout << p[0].x << ' ' << p[0].y << '\n';
		return 0;
	}

	ll mnLen = 1e18;
	int x, y;
	for (auto& i : p) {
		ll nLen = -1;
		for (auto& j : p) {
			nLen = max(nLen, (i.x - j.x) * (i.x - j.x) + (i.y - j.y) * (i.y - j.y));
		}

		if (mnLen > nLen) {
			mnLen = nLen;
			x = i.x, y = i.y;
		}
	}
	
	cout << x << ' ' << y;

	return 0;
}