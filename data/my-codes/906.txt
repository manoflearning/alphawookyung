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

int n, a[1005], visited[1005];

void input() {
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
    }
}

int nxt(int idx, int b) {
    if (b) return (idx < n ? idx + 1 : 1);
    else return (1 < idx ? idx - 1 : n);
}

void f() {
    int x = 1;
    for (int i = 0; i < n; i++) {
        cout << x << ' ';
        
        if (i == n - 1) break;
        
        visited[x] = 1;
        
        int y = fabs(a[x]), b = (a[x] > 0);
        
        for (int j = 0; j < y; j++) {
            x = nxt(x, b);
            while (visited[x]) x = nxt(x, b);
        }
    }
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	f();
	
	return 0;
}
