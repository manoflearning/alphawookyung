#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")

#include <bits/stdc++.h>
using namespace std;

int n, k;
int a[101010], b[101010];

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    cin >> n >> k;
    for (int i = 1; i <= n; i++) cin >> a[i];
    for (int i = 1; i <= n - 1; i++)
        b[i] = a[i + 1] - a[i];

    vector<vector<int>> dp(2, vector<int>(101010));
    
    dp[1 & 1][1] = b[1];
    for (int i = 2; i <= n - 1 - 2 * (k - 1); i++) {
        dp[1 & 1][i] = min(b[i], dp[1 & 1][i - 1]);
    }
    for (int i = 2; i <= k; i++) {
        int bit = (i & 1);
        dp[bit][2 * i - 1] = b[2 * i - 1] + dp[!bit][2 * i - 3];
        for (int j = 2 * i; j <= n - 1 - 2 * (k - i); j++) {
            dp[bit][j] = min(b[j] + dp[!bit][j - 2], dp[bit][j - 1]);
        }
    }
    
    cout << dp[k & 1][n - 1];
}