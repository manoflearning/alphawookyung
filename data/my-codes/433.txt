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
const int MAX = 2500;

string s;
int len1[MAX + 5], len2[MAX + 5];
int dp[MAX + 5];

int f(int x) {
    int& ret = dp[x];
    if(ret != -1) return ret;
    if(x == s.size()) return ret = 0;

    ret = INF;
    for(int i = x; i < s.size(); i++) {
        if(len1[i] >= i - x) ret = min(ret, f(min(i + len1[i], 2 * i - x) + 1));
        if(len2[i] >= i - x + 1) ret = min(ret, f(min(i + len2[i], 2 * i - x + 1) + 1));
    }
    return ret += 1;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    cin>>s;

    for(int i=0; i<s.size(); i++) {
        for(int j=0; 0<=i-j && i+j<s.size(); j++) {
            if(s[i-j] == s[i+j]) len1[i] = j;
            else break;
        }
        for(int j=1; 0<=i-j+1 && i+j<s.size(); j++) {
            if(s[i-j+1] == s[i+j]) len2[i] = j;
            else break;
        }
    }

    memset(dp, -1, sizeof(dp));

    cout<<f(0);

    return 0;
}