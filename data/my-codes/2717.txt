#define ll long long
const int b1 = 1e4, b2 = 1e9;
class Solution {
public:
    string categorizeBox(ll a, ll b, ll c, ll d) {
        int Heavy = 0, Bulky = 0;
        if (a >= b1 || b >= b1 || c >= b1 || d >= b1) Bulky = 1;
        if (a * b * c >= b2) Bulky = 1;
        if (d >= 100) Heavy = 1;
        
        if (Bulky && Heavy) return "Both";
        else if (Bulky) return "Bulky";
        else if (Heavy) return "Heavy";
        else return "Neither";
    }
};