#include <iostream>
#include <vector>
#include <algorithm>
#define ll long long
using namespace std;

struct point {
	ll x, y, px, py, n;
	point() : point(0, 0, 0, 0, 0) {}
	point(int sx, int sy, int sn) : point(sx, sy, 0, 0, sn) {}
	point(int sx, int sy, int spx, int spy, int sn) : x(sx), y(sy), px(spx), py(spy), n(sn) {}
	bool operator <(point& O) {
		if (O.px * py != px * O.py) return O.px * py < px * O.py;
		if (y != O.y) return y < O.y;
		return x < O.x;
	}
};

vector<point> all;
int level[1005];

ll ccw(point& a, point& b, point& c);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	for (int i = 0; i < N; i++) {
		int x, y;
		cin >> x >> y;
		all.push_back(point(x, y, i));
	}

	for (int f = 1; ; f++) {
		vector<point> p;
		for (int i = 0; i < all.size(); i++) {
			if (level[i] != 0) continue;
			p.push_back(point(all[i].x, all[i].y, all[i].n));
		}

		if (p.size() < 3) break;

		sort(p.begin(), p.end());

		for (int i = 1; i < p.size(); i++) {
			p[i].px = p[i].x - p[0].x;
			p[i].py = p[i].y - p[0].y;
		}

		sort(p.begin() + 1, p.end());

		vector<int> st;
		st.push_back(0);
		st.push_back(1);

		for (int next = 2; next < p.size(); next++) {
			while (st.size() >= 2) {
				int first = st.back();
				st.pop_back();
				int second = st.back();
				if (ccw(p[second], p[first], p[next]) > 0) {
					st.push_back(first);
					break;
				}
			}
			st.push_back(next);
		}

		if (st.size() == 2) break;

		for (int x : st) level[p[x].n] = f;
	}

	for (int i = 0; i < N; i++)
		cout << level[i] << ' ';

	return 0;
}

ll ccw(point& a, point& b, point& c) {
	return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
}