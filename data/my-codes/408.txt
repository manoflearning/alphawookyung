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

struct cntx {
    ll cnt, x;
};

bool operator<(cntx a, cntx b) {
    if (a.cnt != b.cnt) return a.cnt > b.cnt;
    return a.x > b.x;
}

int l, n;
vl a;
priority_queue<cntx> pq;

void input() {
    cin >> l;
    a.resize(l);
    for (auto& i : a) cin >> i;
    cin >> n;
}

void f() {
    for (auto& i : a) 
        if (i != 0) pq.push({ 0, i });
    
    for (int i = 0; i < a.size() - 1; i++) {
        if (a[i + 1] - a[i] <= 100) {
            for (ll j = a[i] + 1; j < a[i + 1]; j++) {
	            pq.push({ (j - a[i]) * (a[i + 1] - j) - 1, j });
	        }
        }
	    else {
	        for (ll j = a[i] + 1; j <= a[i] + 50; j++) {
	            pq.push({ (j - a[i]) * (a[i + 1] - j) - 1, j });
	        }
	        for (ll j = a[i + 1] - 1; j >= a[i + 1] - 50; j--) {
	            pq.push({ (j - a[i]) * (a[i + 1] - j) - 1, j });
	        }
	    }
	}
	
	for (int i = 0; i < n; i++) {
	    if (!pq.empty()) {
	        cout << pq.top().x << ' ';
	        pq.pop();
	    }
	    else {
	        cout << ++a.back() << ' ';
	    }
	}
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	a.push_back(0);
	sort(a.begin(), a.end());
	
	f();
	
	return 0;
}
