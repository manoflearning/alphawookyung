#include <bits/stdc++.h>
using namespace std;

int query(int u, int v) {
    cout << "? " << u << ' ' << v << endl;
    int ret; cin >> ret;
    return ret;
}

int main() {
    int tc; cin >> tc;
    while (tc--) {
        int n; cin >> n;

        int u = 1, v = 2;
        int g = query(u, v);
        for (int i = 3; i <= n; i++) {
            int res1 = query(u, i);
            int res2 = query(v, i);

            if (res1 >= max(res2, g)) {
                g = res1;
                v = i;
            }
            else if (res2 >= max(res1, g)) {
                g = res2;
                u = i;
            }
        }

        cout << "! " << u << ' ' << v << endl;
        int res; cin >> res;
    }

    return 0;
}