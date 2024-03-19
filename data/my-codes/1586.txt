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

const int MAXK = 1e4;

int n, k, d;
int a[MAXK + 5], b[MAXK + 5], c[MAXK + 5];

void input() {
    cin >> n >> k >> d;
    for (int i = 0; i < k; i++) {
        cin >> a[i] >> b[i] >> c[i];
    }
}

ll f(int idx) {
    ll ret = 0;
    for (int i = 0; i < k; i++) {
        if (idx >= a[i])
            ret += (min(b[i], idx) - a[i]) / c[i] + 1;
    }
    return ret;
}

int bs(int l, int r) {
    while (l < r) {
        int mid = (l + r) / 2;
        
        if (f(mid) >= d) r = mid;
        else l = mid + 1;
    }
    return l;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	cout << bs(1, n);
	
	return 0;
}
