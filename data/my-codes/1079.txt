#include <bits/stdc++.h>
#include "evolution.h"
using namespace std;
#define sz(x) (int)(x).size()

int n = 1, par[505050];
vector<int> t[505050];
int dp[505050], mxdep[505050];

void init() {}

void observation(int p) {
    assert(p <= n);
    int v = ++n;
    par[v] = p;
    t[p].push_back(v);
    
    if (sz(t[p]) >= 4) return;

    while (1) {
        int isChanged = 0;
        
        int x = -1, y = -1, z = -1;
        for (auto& i : t[p]) {
            if (dp[p] < dp[i]) {
                dp[p] = dp[i];
                isChanged = 1;
            }
            if (x <= mxdep[i]) {
                z = y, y = x, x = mxdep[i];
            }
            else if (y <= mxdep[i]) z = y, y = mxdep[i];
            else if (z <= mxdep[i]) z = mxdep[i];
        }

        y++, z++;
        if (x < y) {
            swap(x, y);
            if (y < z) swap(y, z);
        }
        if (dp[p] < x + y) {
            dp[p] = x + y;
            isChanged = 1;
        }
        if (mxdep[p] < x) {
            mxdep[p] = x;
            isChanged = 1;
        }
        if (dp[p] < mxdep[p]) {
            dp[p] = mxdep[p];
            isChanged = 1;
        }
            
        if (!isChanged) break;

        p = par[p], v = par[v];
        if (p < 1) break;
    }
}

int analyze(int r) { return dp[r]; }