#include <bits/stdc++.h>
using namespace std;
#define pii pair<int, int>
#define fr first
#define sc second
#define sz(x) (int)(x).size()

int n, k;
pii a[1010101];
queue<int> L, R;

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    cin >> n >> k;
    for (int i = 1; i <= n; i++) cin >> a[i].fr;
    for (int i = 1; i <= n; i++) cin >> a[i].sc;

    for (int i = k - 1; i >= 1; i--) L.push(i);
    for (int i = k + 1; i <= n; i++) R.push(i);

    int l = a[k].fr - a[k].sc, r = a[k].fr + a[k].sc;
    
    vector<int> ans(1, k);
    while (1) {
        int bit = 0;
        while (sz(L) && l <= a[L.front()].fr) {
            int idx = L.front();
            L.pop();

            ans.push_back(idx);
            l = min(l, a[idx].fr - a[idx].sc);
            r = max(r, a[idx].fr + a[idx].sc);
            bit = 1;
        }
        while (sz(R) && a[R.front()].fr <= r) {
            int idx = R.front();
            R.pop();

            ans.push_back(idx);
            l = min(l, a[idx].fr - a[idx].sc);
            r = max(r, a[idx].fr + a[idx].sc);
            bit = 1;
        }
        
        if (!bit) break;
    }

    sort(ans.begin(), ans.end());

    for (auto& i : ans) cout << i << ' ';
    
    return 0;
}