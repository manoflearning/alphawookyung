class Solution {
public:
    int miceAndCheese(vector<int>& reward1, vector<int>& reward2, int k) {
        vector<pair<int, int>> arr;
        for (int i = 0; i < reward1.size(); i++) {
            arr.push_back({ reward1[i], reward2[i] });
        }
        
        sort(arr.begin(), arr.end(), [&](const pair<int, int>& p1, const pair<int, int>& p2) {
            return p1.first - p1.second > p2.first - p2.second; 
        });
        
        int ret = 0;
        for (int i = 0; i < arr.size(); i++) {
            if (i < k) ret += arr[i].first;
            else ret += arr[i].second;
        }
        
        return ret;
    }
};