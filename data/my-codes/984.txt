#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <cmath>
#define pii pair<int, int>
using namespace std;

const int INF = 1e9 + 7;

vector<pii> p;
set<pii> s;

int getDist(pii a, pii b);
pii swp(pii a);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	for (int i = 0; i < N; i++) {
		int x, y;
		cin >> x >> y;
		p.push_back({ x, y });
	}

	sort(p.begin(), p.end());

	s.insert(swp(p[0]));
	s.insert(swp(p[1]));

	int ans = getDist(p[0], p[1]);
	
	int last = 0;
	for (int i = 2; i < N; i++) {
		while (last < i) {
			int dx = p[i].first - p[last].first;
			if (dx * dx <= ans) break;
			s.erase(swp(p[last++]));
		}

		int limit = sqrt(ans);

		auto lo = s.lower_bound({ p[i].second - limit, -INF });
		auto up = s.upper_bound({ p[i].second + limit, INF });

		for (auto it = lo; it != up; it++) {
			ans = min(ans, getDist(swp(*it), p[i]));
		}

		s.insert(swp(p[i]));
	}

	cout << ans << '\n';

	return 0;
}

int getDist(pii a, pii b) {
	int X = a.first - b.first, Y = a.second - b.second;
	return X * X + Y * Y;
}

pii swp(pii a) {
	return { a.second, a.first };
}