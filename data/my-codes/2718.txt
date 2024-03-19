class Solution {
public:
    int xorBeauty(vector<int>& nums) {
        int n = nums.size();
        
        int ret = 0;
        for (auto& i : nums) ret ^= i;
        
        return ret;
    }
};