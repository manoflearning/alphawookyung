#include <bits/stdc++.h>
using namespace std;

const int MAX = 1000;

int n;
vector<int> a, ans;
int cnt[MAX + 1], cnt2;

void del(int v) {
    cnt[v]--;
    if (!cnt[v]) cnt2--;
    ans.push_back(v);
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    cin >> n;
    a.resize(n);
    for (auto& i : a) {
        cin >> i;
        if (!cnt[i]) cnt2++;
        cnt[i]++;
    }
    
    for (int i = 0; i < n; i++) {
        if (3 <= cnt2) {
            for (int j = 0; j <= MAX; j++) {
                if (!ans.empty() && ans.back() + 1 == j) continue;
                if (cnt[j]) { del(j); break; }
            }
        }
        else if (cnt2 == 2) {
            int u = -1, v = -1;
            for (int j = 0; j <= MAX; j++) {
                if (cnt[j]) {
                    if (u == -1) u = j;
                    else v = j;
                }
            }

            if (u + 1 == v || (!ans.empty() && ans.back() + 1 == u)) del(v);
            else del(u);
        }
        else if (cnt2 == 1) {
            for (int j = 0; j <= MAX; j++) {
                if (cnt[j]) { del(j); break; }
            }
        }
    }

    for (auto& i : ans)
        cout << i << ' ';
}