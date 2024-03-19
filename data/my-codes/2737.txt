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

int isPa(string s){
    for(int i=0; i<=s.size()/2; i++){
        if(s[i]!=s[s.size()-1-i]) return 0;
    }
    return 1;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    ll res=0;
    for(int i=100; i<=999; i++){
        for(int j=i; j<=999; j++){
            if(isPa(to_string(i*j))) res=max<ll>(res, i*j);
        }
    }
    
    cout<<res;
    
    return 0;
}
