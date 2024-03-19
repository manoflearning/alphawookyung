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

int n, m;
vector<int> k[200005];

void input() {
    cin >> n >> m;
    
    for (int i = 0; i < m; i++) {
        int a; cin >> a;
        k[i].resize(a);
        
        for (auto& j : k[i]) cin >> j;
    }
}

int f() {
    int ret = 1;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k[i].size() - 1; j++) {
            ret &= (k[i][j] >= k[i][j + 1]);
        }
    }
    
    return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	if (f()) cout << "Yes";
	else cout << "No";
	
	return 0;
}
