class Solution {
public:
    vector<vector<int>> findMatrix(vector<int>& nums) {
        vector<vector<int>> ret;
        vector<set<int>> vi;
        
        for (auto& x : nums) {
            int bit = 0;
            for (int i = 0; i < ret.size(); i++) {
                if (vi[i].count(x)) continue;
                ret[i].push_back(x);
                vi[i].insert(x);
                bit = 1;
                break;
            }
            
            if (bit) continue;
            
            ret.resize(ret.size() + 1);
            vi.resize(vi.size() + 1);
            ret.back().push_back(x);
            vi.back().insert(x);
        }
        
        return ret;
    }
};