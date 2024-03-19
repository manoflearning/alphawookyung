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
 
pii p[105];
 
bool isLine(pii& a, pii& b, pii& c) {
    if((b.first - a.first) * (c.second - a.second) - (c.first - a.first) * (b.second - a.second)) return 0;
    else return 1;
}
 
int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    int n; cin>>n;
 
    for(int i=0; i<n; i++)
        cin>>p[i].first>>p[i].second;
    
    int b = 0;
 
    for(int i=0; i<n; i++) {
        for(int j=i+1; j<n; j++) {
            for(int k=j+1; k<n; k++) {
                if(isLine(p[i], p[j], p[k])) b = 1;
            }
        }
    }
 
    if(b) cout<<"Yes";
    else cout<<"No";
 
    return 0;
}
