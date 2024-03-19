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

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    int tc; cin >> tc;
    while (tc--) {
        int n; cin >> n;

        vector<int> a(n), cnt(n + 1);
        for (int i = 0; i < n; i++) {
            cin >> a[i];
            cnt[a[i]]++;
        }

        int mx = -1;
        vector<int> mxa;
        for (int i = 1; i <= n; i++) {
            if (mx < cnt[i]) {
                mx = cnt[i];
                mxa.clear();
            }
            if (mx == cnt[i]) {
                mxa.push_back(i);
            }
        }
        
        cout << (n - mxa.size()) / (mx - 1) - 1 << '\n';
    }

    return 0;
}