class Solution {
public:
    int captureForts(vector<int>& forts) {
        int n = forts.size();
        vector<int> a(n + 1);
        for (int i = 1; i <= n; i++)
            a[i] = forts[i - 1];
        
        int ret = 0;
        int cnt = 0, bit = 0;
        for (int i = 1; i <= n; i++) {
            if (a[i] == 1) cnt = 0, bit = 1;
            if (a[i] == 0) cnt += bit;
            if (a[i] == -1) ret = max(ret, cnt), cnt = bit = 0;
        }
        
        cnt = 0, bit = 0;
        for (int i = n; i >= 1; i--) {
            if (a[i] == 1) cnt = 0, bit = 1;
            if (a[i] == 0) cnt += bit;
            if (a[i] == -1) ret = max(ret, cnt), cnt = bit = 0;
        }
        
        return ret;
    }
};