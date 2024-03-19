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

int n;
char a[55][55];

int f() {
    int ret = 1;
    
    for (int i = 0; i < n; i++) {
        int my = 1, mx = 1;
        for (int j = 0; j < n - 1; j++) {
            my++, mx++;
            if (a[i][j] != a[i][j + 1]) my = 1;
            if (a[j][i] != a[j + 1][i]) mx = 1;
            
            ret = max({ ret, my, mx});
        }
    }
    
    return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	cin >> n;
	
	for (int i = 0; i < n; i++) {
	    for (int j = 0; j < n; j++) {
	        cin >> a[i][j];
	    }
	}
	
	int ans = 0;
	
	for (int y = 0; y < n; y++) {
	    for (int x = 0; x < n; x++) {
	        for (int k = 0; k < 4; k++) {
	            int ny = y + dy[k], nx = x + dx[k];
	            
	            if (a[y][x] != a[ny][nx]) {
	                swap(a[y][x], a[ny][nx]);
	                
	                ans = max(ans, f());
	                
	                swap(a[y][x], a[ny][nx]);
	            }
	        }
	    }
	}
	
	cout << ans;
	
	return 0;
}
