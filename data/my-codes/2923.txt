#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define all(c) (c).begin(), (c).end()
#define sz(x) (int)(x).size()
#define pll pair<ll, ll>

const ll INF = 1e18;
const int MAXN = 101010;

const pll e = { 0, INF };
struct LiChaoTree {
    ll f(pll l, ll x) { return l.first * x + l.second; }
    struct Node {
        ll nl, nr; int l, r;
        pll line;
    };
    vector<Node> t;
    void build(ll nlb, ll nub) {
        t.clear();
        t.push_back({ nlb, nub, -1, -1, e });
    }
    void modify(pll val, int n = 0) {
        ll nl = t[n].nl, nr = t[n].nr;
        ll mid = (nl + nr) >> 1;

        pll llow = t[n].line, lhigh = val;
        if (f(llow, nl) >= f(lhigh, nl)) swap(llow, lhigh);

        if (f(llow, nr) <= f(lhigh, nr)) {
            t[n].line = llow;
            return;
        }
        else if (f(llow, mid) < f(lhigh, mid)) {
            t[n].line = llow;
            if (t[n].r == -1) {
                t[n].r = sz(t);
                t.push_back({ mid + 1, nr, -1, -1, e });
            }
            modify(lhigh, t[n].r);
        }
        else if (f(llow, mid) >= f(lhigh, mid)) {
            t[n].line = lhigh;
            if (t[n].l == -1) {
                t[n].l = sz(t);
                t.push_back({ nl, mid, -1, -1, e });
            }
            modify(llow, t[n].l);
        }
    }
    ll query(ll x, int n = 0) {
        if (n == -1) return e.second;
        ll nl = t[n].nl, nr = t[n].nr;
        ll mid = (nl + nr) >> 1;

        ll ret = f(t[n].line, x);
        if (x <= mid) ret = min(ret, query(x, t[n].l));
        else ret = min(ret, query(x, t[n].r));
        return ret;
    }
}lct;

int n, m;
ll r[MAXN], d[MAXN];
ll p[MAXN], w[MAXN];

void input() {
    cin >> n;
    for (int i = 1; i <= n; i++)
        cin >> r[i] >> d[i];
    cin >> m;
    for (int i = 1; i <= m; i++) cin >> p[i];
    for (int i = 1; i <= m; i++) cin >> w[i];
}

ll solve() {
    ll ret = 0;
    for (int i = 1; i <= n; i++) {
        ret += p[1] * (d[i] - r[i]);
    }

    lct.build(0, (ll)1e11);

    for (int i = 1; i <= m; i++) {
        lct.modify({ p[i], w[i] - w[1] });
    }

    for (int i = 1; i <= n - 1; i++) {
        ret += lct.query(r[i + 1] - d[i]);
    }

    return ret;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    int tc; cin >> tc;
    for (int tt = 1; tt <= tc; tt++) {
        cout << "Case #" << tt << '\n';

        input();

        cout << solve() << '\n';

        cout << flush;
    }
}