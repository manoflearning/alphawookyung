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

const int MAXN = 1e6;

int n, m;
ll a[MAXN + 5], mx;

void input() {
    cin >> n >> m;
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        mx = max(mx, a[i]);
    }
}

int bs(int l, int r) {
    while (l < r) {
        int mid = (l + r + 1) / 2;
        
        ll sum = 0;
        for (int i = 0; i < n; i++) {
            sum += max<ll>(0, a[i] - mid);
        }
        
        if (sum < m) r = mid - 1;
        else l = mid;
    }
    
    return l;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	cout << bs(0, mx - 1);
	
	return 0;
}
