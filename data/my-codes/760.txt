#include <iostream>
#include <set>
#include <stack>
#include <vector>
#include <algorithm>
#define pii pair<int, int>
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

set<pii> s;

ll ccw(point& a, point& b, point& c);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N, Px, Py;
	cin >> N >> Px >> Py;

	s.insert({ Px, Py });
	for (int i = 0; i < N; i++) {
		int x, y;
		cin >> x >> y;
		s.insert({ x, y });
	}

	int cnt = 0;
	while (s.size() >= 3) {
		vector<point> p;

		for (auto& i : s) {
			p.push_back(point(i.first, i.second));
		}

		sort(p.begin(), p.end());

		for (int i = 1; i < s.size(); i++) {
			p[i].px = p[i].x - p[0].x;
			p[i].py = p[i].y - p[0].y;
		}

		sort(p.begin() + 1, p.end());
		
		stack<int> st;
		st.push(0);
		st.push(1);

		for (int next = 2; next < s.size(); next++) {
			while (st.size() >= 2) {
				int first = st.top();
				st.pop();
				int second = st.top();
				if (ccw(p[second], p[first], p[next]) > 0) {
					st.push(first);
					break;
				}
			}
			st.push(next);
		}

		bool YES = true;
		while (!st.empty()) {
			point c = p[st.top()];
			st.pop();
			s.erase({ c.x, c.y });
			if (c.x == Px && c.y == Py) {
				YES = false;
				break;
			}
		}

		if (YES) cnt++;
		else break;
	}

	cout << cnt << '\n';

	return 0;
}

ll ccw(point& a, point& b, point& c) {
	return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}