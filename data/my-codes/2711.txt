class Solution {
public:
    int longestSemiRepetitiveSubstring(string s) {
        int n = s.size();
        int ans = 1;
        for (int i = 0; i < n; i++) {
            int res = 0;
            for (int j = i + 1; j < n; j++) {
                if (s[j - 1] == s[j]) res++;
                if (res <= 1) ans = max(ans, j - i + 1);
            }
        }
        return ans;
    }
};
