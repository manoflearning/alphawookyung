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

struct ts {
    ll t, s;
    int idx;
};

bool operator<(ts& a, ts& b) {
    if (a.t * b.s != b.t * a.s) 
        return a.t * b.s < b.t * a.s;
    else return a.idx < b.idx;
}

int n;
vector<ts> a;

void input() {
    cin >> n;
    a.resize(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i].t >> a[i].s;
        a[i].idx = i + 1;
    }
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	sort(a.begin(), a.end());
	
	for (auto& i : a) cout << i.idx << ' ';
	
	return 0;
}
