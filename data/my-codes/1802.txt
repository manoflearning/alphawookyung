#include "gap.h"
#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define sz(x) (int)(x).size()
#define all(x) (x).begin(), (x).end()

const ll MX = 1e18;

vector<ll> arr;

ll cal() {
    ll ret = 0;
    for (int i = 0; i + 1 < sz(arr); i++) {
        ret = max(ret, arr[i + 1] - arr[i]);
    }
    return ret;
}

ll T1(int N) {
    ll lb = 0, ub = MX;

    stack<ll> stk;
    while (lb <= ub && sz(arr) + sz(stk) < N) {
        ll mn = -1, mx = -1;
        MinMax(lb, ub, &mn, &mx);
        if (mn == -1 && mx == -1) break;
        arr.push_back(mn);
        if (mn < mx) stk.push(mx);
        lb = mn + 1, ub = mx - 1;
    }
    
    while (!stk.empty()) {
        arr.push_back(stk.top());
        stk.pop();
    }

    return cal();
}

ll T2(int N) {
    ll lb = -1, ub = -1;
    MinMax(0, MX, &lb, &ub);
    
    ll anslb = (ub - lb + N - 2) / (N - 1);
    for (ll l = lb; l <= ub; l += anslb + 1) {
        ll r = min(ub, l + anslb);
        ll mn = -1, mx = -1;
        MinMax(l, r, &mn, &mx);
        if (mn != -1) arr.push_back(mn);
        if (mx != -1 && mx != mn) arr.push_back(mx);
    }

    sort(all(arr));

    return cal();
}

ll findGap(int T, int N) {
    srand(time(NULL));
    if (T == 1) return T1(N);
    if (T == 2) return T2(N);
}
