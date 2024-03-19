#include <bits/stdc++.h>
using namespace std;
#define sz(x) (int)(x).size()

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

    int n; cin >> n;

    int flag = 1;
    for (; flag < n; flag <<= 1);
    
    vector<vector<int>> ans;
    for (int i = flag >> 1; i >= 1; i >>= 1) {
        vector<int> res;
        for (int j = 1; j <= n; j++) {
            if (sz(res) && sz(res) % i == 0) j += i;
                
            if (j > n) break;

            res.push_back(j);
        }
        ans.push_back(res);
    }

    cout << sz(ans) << '\n';
    for (auto& i : ans) {
        cout << sz(i) << ' ';
        for (auto& j : i) 
            cout << j << ' ';
        cout << '\n';
    }

	return 0;
}