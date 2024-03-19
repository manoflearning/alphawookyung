#define ll long long
class Solution {
public:
    long long beautifulSubarrays(vector<int>& nums) {
        ll ret = 0;
        map<int, ll> mp;
        int n = nums.size();
        int bit = 0;
        mp[0]++;
        for (int i = 0; i < n; i++) {
            bit ^= nums[i];
            ret += mp[bit];
            mp[bit]++;
        }
        return ret;
    }
};