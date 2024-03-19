#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

const int MAXN = 1e5 + 5;

int n, q;
int psum1[MAXN], psum2[MAXN], psum3[MAXN];

void input() {
    cin >> n >> q;
    for (int i = 1; i <= n; i++) {
        int x; cin >> x;
        
        psum1[i] = psum1[i - 1] + (x == 1);
        psum2[i] = psum2[i - 1] + (x == 2);
        psum3[i] = psum3[i - 1] + (x == 3);
    }
}

void f() {
    for (int i = 0; i < q; i++) {
        int l, r;
        cin >> l >> r;
        cout << psum1[r] - psum1[l - 1] << ' ';
        cout << psum2[r] - psum2[l - 1] << ' ';
        cout << psum3[r] - psum3[l - 1] << '\n';
    }
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	f();
	
	return 0;
}
