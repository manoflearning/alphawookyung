#include <bits/stdc++.h>
using namespace std;

int flag;
struct Seg {
    vector<int> t;
    void build(int n) {
        for (flag = 1; flag < n; flag <<= 1);
        t.resize(flag << 1);
    }
    void modify(int p) {
        for (t[p += flag - 1] = 1; p > 1; p >>= 1)
            t[p >> 1] = t[p] + t[p ^ 1];
    }
    int query(int l, int r, int n = 1, int nl = 1, int nr = flag) {
        if (r < nl || nr < l) return 0;
        if (l <= nl && nr <= r) return t[n];
        int mid = (nl + nr) >> 1;
        return query(l, r, n << 1, nl, mid) + query(l, r, n << 1 | 1, mid + 1, nr);
    }
}seg;

int n, a[505050], idx[1010101];

int main() {
    cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
    
    cin >> n;
    for (int i = 1; i <= n; i++) cin >> a[i];
    for (int i = 1; i <= n; i++) {
        int x; cin >> x;
        idx[x] = i;
    }

    seg.build(n);

    long long ans = 0;
    for (int i = 1; i <= n; i++) {
        ans += seg.query(idx[a[i]] + 1, n);
        seg.modify(idx[a[i]]);
    }

    cout << ans;
}