#include <bits/stdc++.h>
using namespace std;

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    int tc; cin >> tc;
    for (int tt = 1; tt <= tc; tt++) {
        cout << "Case #" << tt << '\n';

        int n, a, b;
        cin >> n >> a >> b;

        int ans = 0;
        for (int i = 0; a * i <= n; i++) {
            if ((n - a * i) % b == 0) {
                ans = max(ans, i + (n - a * i) / b);
            }
        }

        cout << ans << '\n';

        cout << flush;
    }
}