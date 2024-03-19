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

int cnt[11];

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    string s; cin>>s;

    if(s.size() == 1) {
        if(s[0] == '8') cout<<"Yes";
        else cout<<"No";
        return 0;
    }
    if(s.size() == 2) {
        if((10 * (s[0] - '0') + (s[1] - '0')) % 8 == 0 || (10 * (s[1] - '0') + (s[0] - '0')) % 8 == 0) cout<<"Yes";
        else cout<<"No";
        return 0;
    }

    for(auto& c : s) cnt[c - '0']++;

    int res = 0;

    for(int i=1; i<=9; i++) {
        for(int j=1; j<=9; j++) {
            for(int k=1; k<=9; k++) {
                if((100*i+10*j+k) % 8) continue;
                cnt[i]--, cnt[j]--, cnt[k]--;
                if(cnt[i] >= 0 && cnt[j] >= 0 && cnt[k] >= 0) res = 1;
                cnt[i]++, cnt[j]++, cnt[k]++;
            }
        }
    }

    if(res & 1) cout<<"Yes";
    else cout<<"No";

    return 0;
}