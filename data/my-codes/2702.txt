#define ll long long
#define pll pair<ll, ll>
#define fr first
#define sc second
#define sz(x) (int)(x).size()

class Solution {
public:
    ll maxScore(vector<int>& nums1, vector<int>& nums2, int k) {
        int n = nums1.size();
        
        priority_queue<pll> pq;
        for (int i = 0; i < n; i++) {
            pq.push({ nums2[i], nums1[i] });
        }
        
        ll ret = 0, sum = 0;
        priority_queue<ll> pq2;
        while (!pq.empty()) {
            sum += pq.top().sc;
            pq2.push(-pq.top().sc);
            
            if (sz(pq2) > k) {
                sum += pq2.top();
                pq2.pop();
            }
            
            if (sz(pq2) == k) ret = max(ret, sum * pq.top().fr);
            
            pq.pop();
        }
        
        return ret;
    }
};