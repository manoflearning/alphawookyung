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
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int F, h;
map<string, int> Hash;
vi p;

void init() {
    h = 0;
    Hash.clear();
    p.clear();
}

int find(int x) {
	if (p[x] < 0) return x;
	return p[x] = find(p[x]);
}

void merge(int a, int b) {
	int A = find(a), B = find(b);
	if (A == B) return;
	p[A] += p[B];
	p[B] = A;
}

void f() {
    cin >> F;
    
    p.resize(2 * F + 1, -1);
    
    for (int i = 0; i < F; i++) {
        string a, b;
        cin >> a >> b;
        
        if (Hash.find(a) == Hash.end()) Hash.insert({ a, h++ });
        if (Hash.find(b) == Hash.end()) Hash.insert({ b, h++ });
        
        int aa = Hash.find(a)->second;
        int bb = Hash.find(b)->second;
        
        merge(aa, bb);
        assert(find(aa) == find(bb));
        cout << -p[find(aa)] << '\n';
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
