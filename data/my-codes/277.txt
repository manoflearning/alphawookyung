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

const int MAXN = 1e6 + 5;

int n, k, d, isS;
int dist[MAXN];

void init() {
    memset(dist, -1, sizeof(dist));
}

void input() {
    cin >> n >> k;
}

void digit() {
    for (int i = 0; ; i++) {
        if (pow(10, i) <= n && n < pow(10, i + 1)) {
            d = i;
            return;
        }
    }
}

int f(int n, int a, int b) {
    int aa = n % (int)pow(10, a + 1) / (int)pow(10, a);
    int bb = n % (int)pow(10, b + 1) / (int)pow(10, b);
    
    return n - (aa * pow(10, a) + bb * pow(10, b)) + (aa * pow(10, b) + bb * pow(10, a));
}

void f2() {
    for (int i = 0; i <= d; i++) {
        for (int j = i + 1; j <= d; j++) {
            int ii = n % (int)pow(10, i + 1) / (int)pow(10, i);
            int jj = n % (int)pow(10, j + 1) / (int)pow(10, j);
            
            isS |= (ii == jj);
        }
    }
}

void bfs() {
    queue<int> q;
    
    q.push(n);
    dist[n] = 0;
    
    while(!q.empty()) {
        int v = q.front();
        q.pop();
        
        if (dist[v] == k) continue;
        
        for (int i = 0; i <= d; i++) {
            for (int j = i + 1; j <= d; j++) {
                int u = f(v, i, j);
                if (dist[u] >= 0) continue;
                if (u < pow(10, d)) continue;
                
                q.push(u);
                dist[u] = dist[v] + 1;
            }
        }
    }
}

int f3() {
    for (int i = 1e6; i >= 1; i--) {
        if (isS && dist[i] >= 0) return i;
        if (dist[i] >= 0 && (k - dist[i]) % 2 == 0) return i;
    }
    return -1;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	init();
	
	input();
	
	digit();
	
	if (d == 0 || (d == 1 && n % 10 == 0)) {
	    cout << -1;
	    return 0;
	}
	
	bfs();
	
	f2();
	
	cout << f3();
	
	return 0;
}
