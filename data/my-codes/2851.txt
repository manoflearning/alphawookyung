#include <bits/stdc++.h>
using namespace std;

int n, m;
vector<int> uf(202020, -1);

int find(int x) {
    if (uf[x] < 0) return x;
    return uf[x] = find(uf[x]);
}

void merge(int a, int b) {
    int A = find(a), B = find(b);
    if (A == B) return;
    uf[A] += uf[B];
    uf[B] = A;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    cin >> n >> m;
    int ans = 0;
    for (int i = 0; i < m; i++) {
        int u, v; char x, y;
        cin >> u >> x >> v >> y;
        if (find(u) == find(v)) ans++;
        merge(u, v);
    }

    set<int> s;
    for (int i = 1; i <= n; i++) s.insert(find(i));

    cout << ans << ' ' << s.size() - ans;

	return 0;
}