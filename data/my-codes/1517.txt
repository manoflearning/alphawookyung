#include <iostream>
#include <vector>
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
		if (y != O.py) return y < O.y;
		return x < O.x;
	}
};

vector<point> p;

ll ccw(point& a, point& b, point& c);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int N; cin >> N;

	for (int i = 0; i < N; i++) {
		int x, y;
		cin >> x >> y;
		p.push_back(point(x, y));
	}
	
	if (N == 1) {
		cout << 0 << '\n';
		return 0;
	}
	
	sort(p.begin(), p.end());
	
	for (int i = 1; i < N; i++) {
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
	
	double total = 0;
	int stand = st[0];
	for (int i = 0; i < st.size(); i++) {
		int first = st[(i + 1) % st.size()], second = st[i];
		if (first == stand || second == stand) continue;
		total += fabs(ccw(p[second], p[first], p[stand])) / 2;
	}

	cout << (ll)(total / 50) << '\n';

	return 0;
}

ll ccw(point& a, point& b, point& c) {
	return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
}