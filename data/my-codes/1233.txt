#include <iostream>
using namespace std;

const int MAXV = 1e5;

struct Seg {
	int N, t[2 * MAXV];

	void build() {
		cin >> N;
		for (int i = N; i < N + N; i++) {
			int x; cin >> x;
			t[i] = x & 1;
		}
		for (int i = N - 1; i >= 1; i--) t[i] = t[i << 1] + t[i << 1 | 1];
	}
	void modify(int p, int value) {
		for (t[p += N] = value & 1; p > 1; p >>= 1) t[p >> 1] = t[p] + t[p ^ 1];
	}
	int query(int l, int r) {
		int ret = 0;
		for (l += N, r += N; l < r; l >>= 1, r >>= 1) {
			if (l & 1) ret += t[l++];
			if (r & 1) ret += t[--r];
		}
		return ret;
	}
}seg;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	seg.build();
	
	int M; cin >> M;
	while (M--) {
		int op, a, b;
		cin >> op >> a >> b;
		if (op == 1) seg.modify(a - 1, b);
		if (op == 2) cout << b - a + 1 - seg.query(a - 1, b) << '\n';
		if (op == 3) cout << seg.query(a - 1, b) << '\n';
	}

	return 0;
}