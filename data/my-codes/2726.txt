#define ll long long
#define ull unsigned long long
#define ld long double
#define pii pair<int, int>
#define pll pair<ll, ll>
#define fr first
#define sc second
#define all(c) (c).begin(), (c).end()
#define sz(x) (int)(x).size()

class Solution {
public:
    int maximumLength(string s) {
        vector<vector<int>> psum(26, vector<int>(sz(s) + 5));
        
        int len = 0;
        for (int i = 0; i < sz(s); i++) {
            if (i == 0 || (i > 0 && s[i] == s[i - 1])) len++;
            else {
                psum[s[i - 1] - 'a'][len]++;
                len = 1;
            }
        }
        psum[s[sz(s) - 1] - 'a'][len]++;
        
        int ans = -1;
        for (int i = 0; i < 26; i++) {
            for (int j = sz(s) - 1; j >= 1; j--) {
                if (psum[i][j + 1]) {
                    psum[i][j] = min(3, psum[i][j] + psum[i][j + 1] + 1);
                }
                
                if (psum[i][j] >= 3) ans = max(ans, j);
            }
        }
        
        return ans;
    }
};