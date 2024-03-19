class Solution {
public:
    int findTheLongestBalancedSubstring(string s) {
        int ret = 0;
        
        int cnt0 = 0, cnt1 = 0;
        for (auto& i : s) {
            if (i == '0') {
                if (cnt1) cnt0 = cnt1 = 0;
                cnt0++;
            }
            else {
                cnt1++;
                ret = max(ret, 2 * min(cnt0, cnt1));
            }
        }
        
        return ret;
    }
};