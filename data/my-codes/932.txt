#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vl vector<ll>
#define vvl vector<vector<ll>>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

const int MAX = 90;

int L = 1;
ll k, fibo[MAX], dp[MAX][2];

void init() {
    for (int i = 0; i < MAX; i++) {
        dp[i][0] = dp[i][1] = -1;
    }
    
    fibo[1] = fibo[2] = 1;
    for (int i = 3; i < MAX; i++) {
        fibo[i] = fibo[i - 1] + fibo[i - 2];
    }
}

void input() {
    cin >> k;
}

ll f(int len, int b) {
    ll& ret = dp[len][b];
    if (ret != -1) return ret;
    if (len == 1) return ret = 1;
    
    return ret = f(len - 1, 0) + (b ? 0 : f(len - 1, 1));
}

void f2(int len, int b, ll cnt) {
    if (len == 0) return;
    
    if (cnt <= f(len, 0)) {
        cout << 0;
        f2(len - 1, 0, cnt);
    }
    else {
        cout << 1;
        f2(len - 1, 1, cnt - f(len, 0));
    }
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
    
	init();
	
	input();
	
	for (int i = 1; fibo[i] < k; i++) {
	    k -= fibo[i], L = i + 1;
	}
	
	cout << 1;
	f2(L - 1, 1, k);
	
	return 0;
}
