#include <bits/stdc++.h>
using namespace std;
#define ll long long

struct Point { ll x, y; };

int n;
Point st;
vector<Point> p;

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    cin >> n;
    p.resize(n);
    for (auto& i : p)
        cin >> i.x >> i.y;
    cin >> st.x >> st.y;

    for (auto& i : p)
        i.x -= st.x, i.y -= st.y;
    st.x = st.y = 0;

    int u = 0, d = 0, l = 0, r = 0;
    for (auto& i : p) {
        if (i.y >= abs(i.x)) u = 1;
        if (i.y <= -abs(i.x)) d = 1;
        if (abs(i.y) <= i.x) r = 1;
        if (i.x <= -abs(i.y)) l = 1;
    }

    if (!u || !d || !l || !r) cout << "YES";
    else cout << "NO";
}