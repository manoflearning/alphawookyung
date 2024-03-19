#include <iostream>
#include <algorithm>
#define ll long long
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

    ll a, b, c, d;
    cin >> a >> b >> c >> d;
    
    ll ans = 0;
    for (int i = a + b; i <= b + c; i++) {
        ll one = i - a, two = i - b;
        if (one > c) one = c;
        if (two < b) two = b;
        ans += (one - two + 1) * (i <= c ? 0LL : (min(i - c, d - c + 1)));
    }
    cout << ans << '\n';

	return 0;
}