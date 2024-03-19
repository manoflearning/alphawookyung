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

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    for(int a=1; a<=1000; a++){
        for(int b=a+1; a+b<=1000; b++){
            int c=1000-a-b;
            if(a*a+b*b==c*c) {
                cout<<a*b*c;
                a=b=1001;
                break;
            }
        }
    }
    
    return 0;    
}
