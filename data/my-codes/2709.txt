class Solution {
public:
    bool isFascinating(int n) {
        string s1 = to_string(n);
        string s2 = to_string(2 * n);
        string s3 = to_string(3 * n);
        vector<int> cnt(10);
        for (auto& i : s1) cnt[i - '0']++;
        for (auto& i : s2) cnt[i - '0']++;
        for (auto& i : s3) cnt[i - '0']++;
        
        bool ans = 1;
        for (int i = 1; i < 10; i++)
            ans &= (cnt[i] == 1);
        ans &= (cnt[0] == 0);
        
        return ans;
    }
};
