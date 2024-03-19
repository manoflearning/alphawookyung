#define ll long long

const int MOD = 1e9 + 7;

class Solution {
public:
    int sumDistance(vector<int>& nums, string s, int d) {
        int n = nums.size();
        
        vector<ll> a;
        for (auto& i : nums) a.push_back(i);
        
        for (int i = 0; i < n; i++) {
            if (s[i] == 'L') a[i] -= d;
            if (s[i] == 'R') a[i] += d;
        }
        
        sort(a.begin(), a.end());
        
        ll sum = a[0];
        ll ans = 0;
        for (int i = 1; i < n; i++) {
            ans = (ans + i * a[i] - sum) % MOD;
            sum += a[i];
        }
        
        return (int)ans;
    }
};
