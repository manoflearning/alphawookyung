#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#define pii pair<int, int>
using namespace std;

int getDist(pii p, int x);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int M, N, L;
	cin >> M >> N >> L;

	vector<int> shot(M);
	vector<pii> ani(N);

	for (int& i : shot) cin >> i;

	for (auto& i : ani) cin >> i.first >> i.second;

	sort(shot.begin(), shot.end());
	sort(ani.begin(), ani.end());

	int ans = 0, last = 0;
	for (int i = 0; i < M - 1; i++) {
		int mid = (shot[i] + shot[i + 1]) / 2;
		
		while (last != ani.size() && ani[last].first <= mid) {
			if (getDist(ani[last], shot[i]) <= L) ans++;

			last++;
		}
	}

	while (last != ani.size()) {
		if (getDist(ani[last], shot.back()) <= L) ans++;

		last++;
	}

	cout << ans << '\n';

	return 0;
}

int getDist(pii p, int x) {
	return fabs(p.first - x) + p.second;
}