#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
#include <cmath>
#define ll long long
using namespace std;

struct point {
	ll x, y, px, py;
	point() : point(0, 0, 0, 0) {}
	point(int sx, int sy) : point(sx, sy, 0, 0) {}
	point(int sx, int sy, int spx, int spy) : x(sx), y(sy), px(spx), py(spy) {}
	bool operator <(point& O) {
		if (py * O.px != px * O.py) return py * O.px < px * O.py;
		if (y != O.y) return y < O.y;
		return x < O.x;
	}
};

int N, L;
vector<point> p;

ll ccw(point& a, point& b, point& c);
double dist(point& a, point& b);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	cout << fixed; cout.precision(0);

	cin >> N >> L;

	for (int i = 0; i < N; i++) {
		int x, y;
		cin >> x >> y;
		p.push_back(point(x, y));
	}

	sort(p.begin(), p.end());

	for (int i = 1; i < N; i++) {
		p[i].px = p[i].x - p[0].x;
		p[i].py = p[i].y - p[0].y;
	}

	sort(p.begin() + 1, p.end());

	stack<int> st;
	st.push(0);
	st.push(1);

	for (int i = 2; i < N; i++) {
		while (st.size() >= 2) {
			int first = st.top();
			st.pop();
			int second = st.top();
			if (ccw(p[second], p[first], p[i]) > 0) {
				st.push(first);
				break;
			}
		}
		st.push(i);
	}

	double ans = 0;
	ans += dist(p[0], p[st.top()]);
	while (st.size() >= 2) {
		int first = st.top();
		st.pop();
		int second = st.top();
		ans += dist(p[first], p[second]);
	}

	ans += 2 * L * 3.14159;

	cout << ans << '\n';

	return 0;
}

ll ccw(point& a, point& b, point& c) {
	return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

double dist(point& a, point& b) {
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}