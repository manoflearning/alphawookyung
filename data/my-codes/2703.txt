class Solution {
public:
    long long minOperations(vector<int>& nums1, vector<int>& nums2, int k) {
        if (k == 0) {
            int n = nums1.size();
            for (int i = 0; i < n; i++) {
                if (nums1[i] != nums2[i]) return -1;
            }
            return 0;
        }
        
        long long cnt = 0, sum = 0;
        
        int n = nums1.size();
        for (int i = 0; i < n; i++) {
            long long diff = nums1[i] - nums2[i];
            if (diff % k) return -1;
            sum += diff;
            cnt += abs(diff) / k;
        }
        
        if (sum) return -1;
        
        return cnt / 2;
    }
};