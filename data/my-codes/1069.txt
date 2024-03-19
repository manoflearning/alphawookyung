#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long

struct wv {
    ll w, v;
    bool operator<(const wv& rhs) const {
        return v * rhs.w > w * rhs.v;
    }
};

ll gcd(ll a, ll b) {
	if (b == 0) return a;
	else return gcd(b, a % b);
}

ll n, m;
vector<wv> a;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> n >> m;
    a.resize(n);
    for (auto& i : a) cin >> i.w >> i.v;

    sort(a.begin(), a.end());

    ll sum = 0, ans = 0;
    for (auto& i : a) {
        if (sum + i.w >= m) {
            ll x = ans * i.w + (m - sum) * i.v, y = i.w;
            ll g = gcd(x, y);
            cout << x / g << '/' << y / g;
            break;
        }

        sum += i.w;
        ans += i.v;
    }

	return 0;
}