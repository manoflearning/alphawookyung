class Solution {
public:
    string oddString(vector<string>& words) {
        vector<vector<int>> a;
        for (int i = 0; i < words.size(); i++) {
            vector<int> res;
            for (int j = 0; j + 1 < words[i].size(); j++) {
                res.push_back(words[i][j + 1] - words[i][j]);
            }
            a.push_back(res);
        }
        
        string ans;
        for (int i = 0; i + 1 < words.size(); i++) {
            if (a[i] != a[i + 1]) {
                if ((i > 0 && a[i] != a[0]) || (i == 0 && a[i] != a.back())) ans = words[i];
            }
        }
        
        if (ans.empty()) ans = words.back();
        
        return ans;
    }
};