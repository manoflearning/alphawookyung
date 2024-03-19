#define ll long long

class Solution {
public:    
    ll gcd(ll a, ll b) {
        if (b == 0) return a;
        else return gcd(b, a % b);
    }

    ll lcm(ll a, ll b) {
        return a * b / gcd(a, b);
    }

    int minimizeSet(ll d1, ll d2, ll cnt1, ll cnt2) {
        ll lc = lcm(d1, d2);
        
        ll l = 2, r = 2e9;
        while (l < r) {
            ll mid = (l + r) >> 1;
            
            ll both = mid - mid / d1 - mid / d2 + mid / lc;
            ll only2 = mid / d1 - mid / lc;
            ll only1 = mid / d2 - mid / lc;
            
            if (only1 > cnt1) only1 = cnt1;
            if (only2 > cnt2) only2 = cnt2;
            
            if (only1 + only2 + both >= cnt1 + cnt2) r = mid;
            else l = mid + 1;
        }
        return (int)l;
    }
};