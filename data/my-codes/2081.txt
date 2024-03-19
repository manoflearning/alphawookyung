#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

struct dv {
	int d, bit, v;
	bool operator<(dv& rhs) {
		return d < rhs.d;
	}
};

bool operator<(dv a, dv b) {
	return a.d > b.d;
}

int N, K;
int T[15][15];
int dist[15][1 << 10];

int dijkstra() {
	for (int i = 0; i < N; i++)
		fill(dist[i], dist[i] + (1 << 10), INF);

	priority_queue<dv> pq;
	pq.push({ 0, 1 << K, K });
	dist[K][1 << K] = 0;

	while (!pq.empty()) {
		int d = pq.top().d, bit = pq.top().bit, now = pq.top().v;
		pq.pop();

		if (bit == (1 << N) - 1) return d;

		for (int i = 0; i < N; i++) {
			int nbit = bit + (bit & (1 << i) ? 0 : (1 << i));
			if (d + T[now][i] < dist[i][nbit]) {
				dist[i][nbit] = d + T[now][i];
				pq.push({ dist[i][nbit], nbit, i });
			}
		}
	}
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N >> K;
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			cin >> T[y][x];
		}
	}

	cout << dijkstra();

	return 0;
}