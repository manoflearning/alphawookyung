#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

const int INF = 1e9 + 7;

struct sgtr {
	int mxL, mxR, mx, all;
};

int U, V;
int flag;
vector<sgtr> t;

sgtr f(sgtr left, sgtr right);
sgtr query(int L, int R, int n, int nL, int nR);
void update(int p, int val);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N, Q; 
	cin >> N >> Q >> U >> V;

	for (flag = 1; flag < N; flag *= 2);
	t.resize(2 * flag);

	for (int i = flag; i < flag + N; i++) {
		int x; cin >> x;
		t[i] = { U * x + V, U * x + V, U * x + V, U * x + V };
	}

	for (int i = flag - 1; i >= 1; i--) {
		t[i] = f(t[i << 1], t[i << 1 | 1]);
	}

	while (Q--) {
		int c, a, b;
		cin >> c >> a >> b;

		if (c == 0) cout << query(a, b, 1, 1, flag).mx - V << '\n';
		else update(a, b);
	}

	return 0;
}

sgtr f(sgtr left, sgtr right) {
	sgtr ret;
	ret.mxL = max(left.mxL, left.all + right.mxL);
	ret.mxR = max(right.mxR, left.mxR + right.all);
	ret.mx = max({ left.mx, right.mx, left.mxR + right.mxL });
	ret.all = left.all + right.all;
	return ret;
}

sgtr query(int L, int R, int n, int nL, int nR) {
	if (R < nL || nR < L) return { -INF, -INF, -INF };
	if (L <= nL && nR <= R) return t[n];

	int mid = (nL + nR) >> 1;
	return f(query(L, R, 2 * n, nL, mid), query(L, R, 2 * n + 1, mid + 1, nR));
}

void update(int p, int val) {
	p += flag - 1;
	t[p] = { U * val + V, U * val + V, U * val + V, U * val + V };

	while (p > 1) {
		p >>= 1;
		t[p] = f(t[p << 1], t[p << 1 | 1]);
	}
}