#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

const int INF = 1e9 + 7;

struct Seg {
	int flag;
	vector<int> t;

	void build(int N, int M) {
		for (flag = 1; flag < N; flag <<= 1);
		t.resize(2 * flag);

		for (int i = flag; i < flag + flag; i++) t[i] = INF;
		while (M--) {
			int x; cin >> x;
			t[x + flag - 1] = x;
		}
		for (int i = flag - 1; i >= 1; i--) t[i] = min(t[i << 1], t[i << 1 | 1]);
	}
	void modify(int p) {
		for (t[p += flag - 1] = INF; p > 1; p >>= 1) t[p >> 1] = min(t[p], t[p ^ 1]);
	}
	int query(int L) {
		return query(L, flag, 1, 1, flag);
	}
	int query(int L, int R, int n, int nL, int nR) {
		if (nR < L || R < nL) return INF;
		if (L <= nL && nR <= R) return t[n];

		int mid = (nL + nR) >> 1;
		return min(query(L, R, n << 1, nL, mid), query(L, R, n << 1 | 1, mid + 1, nR));
	}
}seg;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int N, M, K;
	cin >> N >> M >> K;

	seg.build(N, M);

	while (K--) {
		int x; cin >> x;

		int ans = seg.query(x + 1);
		cout << ans << '\n';
		seg.modify(ans);
	}

	return 0;
}