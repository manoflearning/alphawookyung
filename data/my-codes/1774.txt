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

int n, k;
int a[15][15];

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    cin>>n>>k;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            cin>>a[i][j];
        }
    }
    
    for(int y=0; y<n; y++){
        for(int j=0; j<k; j++){
            for(int x=0; x<n; x++){
                for(int i=0; i<k; i++) cout<<a[y][x]<<' ';
            }
            cout<<'\n';
        }
    }
    
    return 0;    
}
