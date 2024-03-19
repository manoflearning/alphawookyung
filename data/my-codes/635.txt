#include <iostream>
#include <vector>
#include <stack>
#include <cmath>
#include <algorithm>
#define ll long long
using namespace std;

struct point {
	ll x, y, px, py;
	point() : point(0, 0, 0, 0) {}
	point(int sx, int sy) : point(sx, sy, 0, 0) {}
	point(int sx, int sy, int spx, int spy) : x(sx), y(sy), px(spx), py(spy) {}
	bool operator <(point& O) {
		if (O.px * py != px * O.py) return O.px * py < px * O.py;
		if (y != O.y) return y < O.y;
		return x < O.x;
	}
};

vector<point> p;

ll ccw(point& a, point& b, point& c);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	cout << fixed; cout.precision(7);

	int C; cin >> C;

	for (int i = 0; i < C; i++) {
		int x, y;
		cin >> x >> y;
		p.push_back(point(x, y));
	}

	sort(p.begin(), p.end());

	for (int i = 1; i < C; i++) {
		p[i].px = p[i].x - p[0].x;
		p[i].py = p[i].y - p[0].y;
	}

	sort(p.begin() + 1, p.end());

	vector<int> st;
	st.push_back(0);
	st.push_back(1);

	for (int next = 2; next < C; next++) {
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

	double len = -1;
	for (int i = 0; i < st.size(); i++) {
		for (int l = i + 1; l < st.size(); l++) {
			auto p1 = p[st[i]], p2 = p[st[l]];
			len = max(len, sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)));
		}
	}

	cout << len << '\n';

	return 0;
}

ll ccw(point& a, point& b, point& c) {
	return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
}