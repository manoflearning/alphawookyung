#define ll long long
#define fr first
#define sc second

class Solution {
public:
    int n, r, k;
    vector<ll> a, psum;
    
    ll get(int idx) {
        return psum[min(n, idx + r)] - psum[max(0, idx - r - 1)];
    }
    
    ll maxPower(vector<int> stations, int R, int K) {
        n = stations.size();
        r = R, k = K;
        a.resize(n + 1), psum.resize(n + 1);
        for (int i = 1; i <= n; i++) {
            a[i] = stations[i - 1];
            psum[i] = a[i] + psum[i - 1];
        }
        
        ll s = 0, e = 1e11;
        while (s < e) {
            ll mid = (s + e + 1) >> 1;
            
            ll sum = 0, res = 0;
            queue<pair<int, int>> q;
            for (int i = 1; i <= n; i++) {
                while (q.size() && q.front().fr < i - r) {
                    sum -= q.front().sc;
                    q.pop();
                }
                
                ll x = mid - (get(i) + sum);
                if (x > 0) {
                    sum += x, res += x;
                    q.push({ min(n, i + r), x });
                }
            }
            
            if (res > k) e = mid - 1;
            else s = mid;
        }
        
        return s;
    }
};