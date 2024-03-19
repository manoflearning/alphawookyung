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

struct point{
    ll x, y;
};

ll a, b, c;
point p[4];
ll w[4];

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    cin>>a>>b>>c;
    cin>>p[0].x>>p[1].x>>p[0].y>>p[1].y;
    
    p[2]={p[0].x, p[1].y};
    p[3]={p[1].x, p[0].y};
    
    for(int i=0; i<4; i++)
        w[i]=a*p[i].x+b*p[i].y+c;
        
    int b1=1, b2=1;
    for(int i=0; i<4; i++){
        b1&=(w[i]<=0);
        b2&=(w[i]>=0);
    }
    
    if(b1 | b2) cout<<"Lucky";
    else cout<<"Poor";
    
    return 0;    
}
