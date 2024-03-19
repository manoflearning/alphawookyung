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
        map<string, int> mp;
        for (int i = 0; i < sz(s); i++) {
            string sub;
            for (int j = i; j < sz(s); j++) {
                if (s[j] != s[i]) break;
                sub.push_back(s[j]);
                mp[sub]++;
            }
        }
        
        int ans = -1;
        for (auto& it : mp) {
            if (it.second >= 3) {
                ans = max(ans, sz(it.first));
            }
        }
        
        return ans;
    }
};