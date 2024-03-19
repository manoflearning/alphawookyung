#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vl vector<ll>
#define vvl vector<vector<ll>>

const int INF = 1e9 + 7;
const int MOD = 1000;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int n;
vi p, d;

void init() {
    p.clear();
    d.clear();
}

int find(int x) {
	if (p[x] < 0) return x;
	
	int pp = p[x];
	p[x] = find(p[x]);
	
	d[x] += d[pp];
	
	return p[x];
}

int dist(int x) {
    find(x);
    return d[x];
}

void merge(int a, int b) {
    int B = find(b);
	if (a == B) return;
	p[a] = B;
	d[a] = (int)fabs(a - b) % MOD + dist(b);
}

void f() {
    cin >> n;
    p.resize(n + 1, -1);
    d.resize(n + 1, 0);
    
    while (1) {
        char c; cin >> c;
        
        if (c == 'O') break;
        
        if (c == 'E') {
            int a; cin >> a;
            cout << dist(a) << '\n';
        }
        
        if (c == 'I') {
            int a, b;
            cin >> a >> b;
            merge(a, b);
        }
    }
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int tc; cin >> tc;
	
	while (tc--) {
	    init();
	    
	    f();
	}
	
	return 0;
}
