#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

const int INF = 1e9 + 7;

struct cv {
	int cost, value;

	bool operator<(cv& rhs) {
		if (value != rhs.value) return value < rhs.value;
		return cost < rhs.cost;
	}
	bool operator<(int rhs) {
		return value < rhs;
	}
};

int N, cost[37], value[37], K;
int have, mid;
vector<cv> L, R;
int minCost[(1 << 16) + 5];

void input();
void f(int idx, int cSum, int vSum);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	if (K == 0) {
		cout << 0;
		return 0;
	}

	if (N == 1) {
		if (K <= value[0]) cout << max(0, cost[0] - have);
		else cout << -1;
		return 0;
	}

	mid = (0 + N - 1) >> 1;

	f(0, 0, 0);
	f(mid + 1, 0, 0);
	
	sort(L.begin(), L.end());
	sort(R.begin(), R.end());

	minCost[R.size() - 1] = R.back().cost;
	for (int i = R.size() - 2; i >= 0; i--) {
		minCost[i] = min(R[i].cost, minCost[i + 1]);
	}

	int ans = INF;
	for (int i = 0; i < L.size(); i++) {
		if (0 < i && L[i - 1].value == L[i].value) continue;

		int left = K - L[i].value;
		int idx = lower_bound(R.begin(), R.end(), left) - R.begin();
		
		if (idx != R.size()) ans = min(ans, L[i].cost + minCost[idx]);
	}

	if (ans == INF) cout << -1;
	else cout << max(0, ans - have);

	return 0;
}

void input() {
	cin >> N;

	for (int i = 0; i < N; i++) cin >> cost[i];
	for (int i = 0; i < N; i++) cin >> value[i];

	cin >> K;

	int cnt; cin >> cnt;
	while (cnt--) {
		int x; cin >> x;
		have += cost[x];
	}
}

void f(int idx, int cSum, int vSum) {
	if (idx == mid || idx == N - 1) {
		if (idx == mid) {
			L.push_back({ cSum, vSum });
			L.push_back({ cSum + cost[idx], vSum + value[idx] });
		}
		else {
			R.push_back({ cSum, vSum });
			R.push_back({ cSum + cost[idx], vSum + value[idx] });
		}
		return;
	}

	f(idx + 1, cSum, vSum);
	f(idx + 1, cSum + cost[idx], vSum + value[idx]);
}