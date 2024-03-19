#include <bits/stdc++.h>
using namespace std;

int n;
vector<string> a;

int main() {
    cin >> n;
    a.resize(n);
    for (auto& i : a) cin >> i;

    int ans = 1e9 + 7;
    for (int bit = 0; bit < (1 << n); bit++) {
        vector<string> b = a;
        for (int i = 0; i < n; i++) {
            if (!(bit & (1 << i))) continue;
            for (auto& j : b[i]) {
                if (j == 'H') j = 'T';
                else j = 'H';
            }
        }

        int res = 0;
        for (int j = 0; j < n; j++) {
            int cnt = 0;
            for (int i = 0; i < n; i++) {
                if (b[i][j] == 'H') cnt++;
            }
            res += min(cnt, n - cnt);
        }

        ans = min(ans, res);
    }

    cout << ans;
}