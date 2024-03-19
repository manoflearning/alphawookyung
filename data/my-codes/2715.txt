class Solution {
public:
    vector<int> topStudents(vector<string>& positive_feedback, vector<string>& negative_feedback, vector<string>& report, vector<int>& student_id, int k) {
        int n = student_id.size();
        vector<pair<int, int>> a(n);
        for (int i = 0; i < n; i++) {
            a[i].second = student_id[i];
        }
        
        set<string> po, ne;
        for (auto& i : positive_feedback) po.insert(i);
        for (auto& i : negative_feedback) ne.insert(i);
        
        for (int i = 0; i < n; i++) {
            string s = report[i];
            int res = 0;
            
            stringstream ss(s);
            string word;
            while (getline(ss, word, ' ')) {
                if (po.count(word)) res += 3;
                if (ne.count(word)) res -= 1;
            }
            
            a[i].first = -res;
        }
        
        sort(a.begin(), a.end());
        
        vector<int> ret(k);
        for (int i = 0; i < k; i++)
            ret[i] = a[i].second;
        
        return ret;
    }
};