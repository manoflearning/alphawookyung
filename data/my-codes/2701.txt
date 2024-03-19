class Solution {
public:
    int getCommon(vector<int>& nums1, vector<int>& nums2) {
        set<int> s;
        for (auto& i : nums1) s.insert(i);
        
        for (auto& i : nums2)
            if (s.count(i)) return i;
        
        return -1;
    }
};