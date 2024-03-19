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

struct lh {
    int l, h;
};

bool operator<(lh a, lh b) {
    return a.l < b.l;
}

int n, mxl, mxr;
vector<lh> a;
stack<int> stk;

void input() {
    cin >> n;
    for (int i = 0; i < n; i++) {
        int l, h;
        cin >> l >> h;
        a.push_back({ l, h });
    }
}

void init() {
    mxl = 0, mxr = n - 1;
    for (int i = 0; i < n; i++) {
        if (a[mxl].h < a[i].h) mxl = i;
    }
    
    for (int i = n - 1; i >= 0; i--) {
        if (a[i].h > a[mxr].h) mxr = i;
    }
}

int f2(int st, int en) {
    int ret = 0;
    
    stk.push(st);
    for (int i = st + 1; i != en; i += (st < en ? 1 : -1)) {
        if (a[stk.top()].h < a[i].h) stk.push(i);
    }
    
    int prv = en;
    while (!stk.empty()) {
        ret += a[stk.top()].h * fabs(a[prv].l - a[stk.top()].l);
        prv = stk.top();
        stk.pop();
    }
    
    return ret;
}

int f() {
    int ret = a[mxl].h * (a[mxr].l - a[mxl].l + 1), prv;
    
    ret += f2(0, mxl);
    ret += f2(n - 1, mxr);
    
    return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	sort(a.begin(), a.end());
	
	init();
	
	cout << f();
	
	return 0;
}
