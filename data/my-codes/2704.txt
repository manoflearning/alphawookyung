#define ll long long

class Solution {
public:
    ll gcd(ll a, ll b) {
        if (b == 0) return a;
        else return gcd(b, a % b);
    }
    bool isPowerOf2(ll x) {
        ll res = 1;
        for (; res < x; res *= 2);
        return res == x;
    }
    bool isReachable(ll x, ll y) {
        return isPowerOf2(gcd(x, y));
    }
};