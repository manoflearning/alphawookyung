class Solution {
public:
    vector<vector<int>> rangeAddQueries(int n, vector<vector<int>>& queries) {
        vector<vector<int>> ans(n, vector<int>(n));
        for (auto& i : queries) {
            ans[i[0]][i[1]]++;
            if (i[2] + 1 < n) ans[i[2] + 1][i[1]]--;
            if (i[3] + 1 < n) ans[i[0]][i[3] + 1]--;
            if (i[2] + 1 < n && i[3] + 1 < n) ans[i[2] + 1][i[3] + 1]++;
        }
        for (int i = 1; i < n; i++) {
            ans[i][0] += ans[i - 1][0];
            ans[0][i] += ans[0][i - 1];
        }
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < n; j++) {
                ans[i][j] += ans[i][j - 1];
                ans[i][j] += ans[i - 1][j];
                ans[i][j] -= ans[i - 1][j - 1];
            }
        }
        return ans;
    }
};