#include <bits/stdc++.h>
using namespace std;

int n, x;
int a[101010], b[101010], mx5 = 0, vi[101010];

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    cin >> n >> x;
    for (int i = 1; i <= n; i++)
        cin >> a[i] >> b[i];

    for (int l = 0; l <= n; l++) {
        int r = n - l;
        if (l * 1000 + r * 5000 <= x) mx5 = max(mx5, r);
    }
    
    priority_queue<pair<int, int>> pq;
    for (int i = 1; i <= n; i++) 
        pq.push({ a[i] - b[i], i });

    int ans = 0;
    while (!pq.empty() && mx5) {
        mx5--;
        auto idx = pq.top().second; 
        pq.pop();
        
        ans += max(a[idx], b[idx]);
        vi[idx] = 1;
    }

    for (int i = 1; i <= n; i++)
        if (!vi[i]) ans += b[i];
    cout << ans;
}