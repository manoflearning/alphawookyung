#include <bits/stdc++.h>
using namespace std;

int n;
int vi[202020];

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    cin >> n;
    for (int v = 1; v <= n; v++) {
        int x; cin >> x;
        if (vi[v]) continue;
        vi[x] = 1;
    }

    int ans = 0;
    for (int v = 1; v <= n; v++)
        if (!vi[v]) ans++;

    cout << ans << '\n';
    for (int v = 1; v <= n; v++)
        if (!vi[v]) cout << v << ' ';
}