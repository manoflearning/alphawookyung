#include <bits/stdc++.h>
using namespace std;
#define all(c) (c).begin(), (c).end()
#define sz(x) (int)(x).size()

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    int tc; cin >> tc;
    for (int tt = 1; tt <= tc; tt++) {    
        cout << "Case #" << tt << '\n';

        int n, d;
        cin >> n >> d;

        vector<int> a;
        int cnt0 = 0;
        for (int i = 0; i < n; i++) {
            int x; cin >> x;
            if (x != 0) a.push_back(x);
            else cnt0++;
        }
        a.push_back(0);

        sort(all(a));

        int stl = -1, str = -1;
        for (int i = 0; i < sz(a); i++) {
            if (a[i] < 0) stl = i;
            if (a[i] > 0) if (str == -1) str = i;
        }

        int ans = 0;
        for (int i = 0; i < sz(a); i++) {
            if (d < abs(a[i])) continue;

            int res = 0;
            if (d <= 2 * abs(a[i])) {
                if (a[i] < 0) res += stl - i + 1;
                if (a[i] > 0) res += i - str + 1;
            }
            if (2 * abs(a[i]) < d) {
                int rd = d - 2 * abs(a[i]);
                if (a[i] < 0) {
                    res += stl - i + 1;
                    res += upper_bound(all(a), rd) - a.begin() - str;
                }
                if (a[i] > 0) {
                    res += i - str + 1;
                    res += stl - (lower_bound(all(a), -rd) - a.begin()) + 1;
                }
            }

            ans = max(ans, res);
        }

        cout << ans + cnt0 << '\n';

        cout << flush;
    }
}