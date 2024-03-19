class Solution {
public:
    bool hasTrailingZeros(vector<int>& nums) {
        for (int i = 0; i < nums.size(); i++) {
            for (int j = i + 1; j < nums.size(); j++) {
                int bit = (nums[i] | nums[j]);
                if (!(bit & 1)) return true;
            }
        }
        return false;
    }
};