#include <iostream>
#include <vector>
#include <cassert>
#define ll long long
using namespace std;

int flag;
vector<ll> sgtr, lazy, cnt;

void update(int L, int R, int n, int nL, int nR);
ll sum(int L, int R, int n, int nL, int nR);
void propagate(int n, int nL, int nR);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;
	
	for (flag = 1; flag < N; flag *= 2);
	sgtr.resize(2 * flag);
	lazy.resize(2 * flag);
	cnt.resize(2 * flag);

	for (int i = flag; i < flag + N; i++) cin >> sgtr[i];
	for (int i = flag - 1; i >= 1; i--)
		sgtr[i] = sgtr[2 * i] + sgtr[2 * i + 1];

	int Q; cin >> Q;
	while (Q--) {
		int q; cin >> q;
		if (q == 1) {
			int L, R;
			cin >> L >> R;
			update(L, R, 1, 1, flag);
		}
		else {
			int x; cin >> x;
			cout << sum(x, x, 1, 1, flag) << '\n';
		}
	}

	return 0;
}

void update(int L, int R, int n, int nL, int nR) {
	propagate(n, nL, nR);
	
	if (R < nL || nR < L) return;
	if (L <= nL && nR <= R) {
		lazy[n] = nL - L + 1;
		cnt[n] = 1;
		propagate(n, nL, nR);
		return;
	}

	int mid = (nL + nR) / 2;
	update(L, R, 2 * n, nL, mid);
	update(L, R, 2 * n + 1, mid + 1, nR);
	sgtr[n] = sgtr[2 * n] + sgtr[2 * n + 1];
}

ll sum(int L, int R, int n, int nL, int nR) {
	propagate(n, nL, nR);

	if (R < nL || nR < L) return 0;
	if (L <= nL && nR <= R) return sgtr[n];

	int mid = (nL + nR) / 2;
	return sum(L, R, 2 * n, nL, mid) + sum(L, R, 2 * n + 1, mid + 1, nR);
}

void propagate(int n, int nL, int nR) {
	if (lazy[n] > 0) {
		ll len = nR - nL + 1;
		if (n < flag) {
			lazy[2 * n] += lazy[n];
			lazy[2 * n + 1] += lazy[n] + cnt[n] * len / 2;
			cnt[2 * n] += cnt[n];
			cnt[2 * n + 1] += cnt[n];
		}
		sgtr[n] += (2 * lazy[n] + cnt[n] * (len - 1)) * len / 2;
		lazy[n] = cnt[n] = 0;
	}
}