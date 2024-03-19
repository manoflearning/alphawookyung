class Solution {
public:
    int destroyTargets(vector<int>& nums, int space) {
        map<int, int> mp;
        
        for (int i = 0; i < nums.size(); i++) {
            mp[nums[i] % space]++;
        }
        
        int ans = 0;
        for (auto& i : mp) 
            ans = max(ans, i.second);
        
        int ret = 1e9 + 7;;
        for (int i = 0; i < nums.size(); i++) {
            if (ans == mp[nums[i] % space]) {
                ret = min(ret, nums[i]);
            }
        }
        
        return ret;
    }
};