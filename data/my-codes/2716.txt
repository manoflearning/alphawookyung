#define ll long long

class Solution {
public:
    const int MOD = 1e9 + 7;
    
    ll powxy(ll x, ll y) { 
        if (y == 0) return 1;
        if (y == 1) return x;
        ll res = powxy(x, y / 2);
        return res * res % MOD * (y & 1 ? x : 1) % MOD;
    }

    int countAnagrams(string s) {
        vector<ll> fac(101010);
        fac[0] = 1;
        for (int i = 1; i < 101010; i++)
            fac[i] = i * fac[i - 1] % MOD;
        
        stringstream ss(s);
 
        string t;
        ll ret = 1;
        while (std::getline(ss, t, ' ')) {
            vector<ll> cnt(26);
            ll sum = t.size();
            for (auto& c : t) {
                cnt[c - 'a']++;
            }
            
            int res = 1;
            res = res * fac[sum] % MOD;
            for (int i = 0; i < 26; i++) {
                if (!cnt[i]) continue;
                res = res * powxy(fac[cnt[i]], MOD - 2) % MOD;
            }
            
            ret = ret * res % MOD;
        }
        
        return ret;
    }
};