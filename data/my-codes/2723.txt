class Solution {
public:
    int maxScore(vector<int>& nums) {
        vector<long long> pos, neg;
        for (auto& i : nums) {
            if (i > 0) pos.push_back(i);
            else neg.push_back(i);
        }
        int ret = 0;
        long long sum = 0;
        for (auto& i : pos) ret++, sum += i;
        sort(neg.rbegin(), neg.rend());
        for (auto& i : neg) {
            sum += i;
            if (sum > 0) ret++;
        }
        return ret;
    }
};