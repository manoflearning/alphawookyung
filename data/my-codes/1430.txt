#include <iostream>
#include <vector>
#include <algorithm>
#define ll long long
#define pii pair<int, int>
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

void convexHull(int N, vector<point>& p, vector<int>& st);
ll ccw(point& a, point& b, point& c);
bool pointCheck(vector<int>& chn, vector<int>& pn, vector<point>& ch, vector<point>& p);
bool lineCheck(vector<int>& chn, vector<int>& pn, vector<point>& ch, vector<point>& p);
bool isCross(point& a, point& b, point& c, point& d);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N, M;
		cin >> N >> M;

		vector<point> bp, wp;
		for (int i = 0; i < N; i++) {
			int x, y;
			cin >> x >> y;
			bp.push_back(point(x, y));
		}

		for (int i = 0; i < M; i++) {
			int x, y;
			cin >> x >> y;
			wp.push_back(point(x, y));
		}

		if (N < M) {
			swap(N, M);
			swap(bp, wp);
		}
		
		if (N == 1 && M == 1) cout << "YES\n";
		else if (M == 1) {
			vector<int> bst, wst;
			convexHull(N, bp, bst);
			wst.push_back(0);

			if (pointCheck(bst, wst, bp, wp)) cout << "NO\n";
			else cout << "YES\n";
		}
		else {
			vector<int> bst, wst;
			convexHull(N, bp, bst);
			convexHull(M, wp, wst);
			
			if (pointCheck(bst, wst, bp, wp)) cout << "NO\n";
			else if (pointCheck(wst, bst, wp, bp)) cout << "NO\n";
			else if (lineCheck(bst, wst, bp, wp)) cout << "NO\n";
			else if (lineCheck(bst, wst, bp, wp)) cout << "NO\n";
			else cout << "YES\n";
		}
	}

	return 0;
}

void convexHull(int N, vector<point>& p, vector<int>& st) {
	sort(p.begin(), p.end());
	
	for (int i = 1; i < N; i++) {
		p[i].px = p[i].x - p[0].x;
		p[i].py = p[i].y - p[0].y;
	}
	
	sort(p.begin() + 1, p.end());

	st.push_back(0);
	st.push_back(1);
	
	for (int next = 2; next < N; next++) {
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
}

ll ccw(point& a, point& b, point& c) {
	ll flag = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
	if (flag > 0) return 1;
	else if (flag == 0) return 0;
	else return -1;
}

bool pointCheck(vector<int>& chn, vector<int>& pn, vector<point>& ch, vector<point>& p) {
	if (chn.size() == 2) {
		int second = chn[0], first = chn[1];
		for (int i = 0; i < pn.size(); i++) {
			if (ccw(ch[second], ch[first], p[pn[i]]) == 0) {
				pii A = { ch[second].x, ch[second].y }, B = { ch[first].x, ch[first].y }, C = { p[pn[i]].x, p[pn[i]].y };
				if (A > B) swap(A, B);
				if (A <= C && C <= B) return true;
			}
		}
		return false;
	}
	
	bool ret = false;
	for (int i = 0; i < pn.size(); i++) {
		bool out = false;
		for (int l = 0; l < chn.size(); l++) {
			int second = chn[l], first = chn[(l + 1) % chn.size()];
			if (ccw(ch[second], ch[first], p[pn[i]]) < 0) out = true;
		}
		if (!out) ret = true;
	}
	return ret;
}

bool lineCheck(vector<int>& chn, vector<int>& pn, vector<point>& ch, vector<point>& p) {
	bool ret = false;
	for (int i = 0; i < pn.size(); i++) {
		int psecond = pn[i], pfirst = pn[(i + 1) % pn.size()];
		for (int l = 0; l < chn.size(); l++) {
			int second = chn[l], first = chn[(l + 1) % chn.size()];
			if (isCross(ch[second], ch[first], p[psecond], p[pfirst])) ret = true;
		}
	}
	return ret;
}

bool isCross(point& a, point& b, point& c, point& d) {
	int ab = ccw(a, b, c) * ccw(a, b, d);
	int cd = ccw(c, d, a) * ccw(c, d, b);
	if (ab == 0 && cd == 0) {
		pii A = { a.x, a.y }, B = { b.x, b.y }, C = { c.x, c.y }, D = { d.x, d.y };
		if (A > B) swap(A, B);
		if (C > D) swap(C, D);
		return (A <= D && C <= B);
	}
	else return (ab <= 0 && cd <= 0);
}