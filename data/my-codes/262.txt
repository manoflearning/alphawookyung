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

string s;

void input() {
    cin >> s;
}

int f(int idx) {
    if (idx == s.size()) return 1;
    
    if (s[idx] == '0') {
        if (idx + 1 < s.size() && s[idx + 1] == '1') return f(idx + 2);
        else return 0;
    }
    else if (s[idx] == '1') {
        for (int i = idx + 1; ; i++) {
            if (i < s.size() && s[i] == '0') continue;
            else {
                if (i - idx <= 2) return 0;
                idx = i;
                break;
            }
        }
        int ret = 0;
        for (int i = idx; ; i++) {
            if (i < s.size() && s[i] == '1') {
                if (i - idx >= 1) ret |= f(i);
            }
            else {
                if (i - idx <= 0) return 0;
                else {
                    ret |= f(i);
                    break;
                }
            } 
        }
        return ret;
    }
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int tc; cin >> tc;
	
	while (tc--) {
	    input();
	    
	    if (f(0)) cout << "YES\n";
	    else cout << "NO\n";
	}
	
	return 0;
}
