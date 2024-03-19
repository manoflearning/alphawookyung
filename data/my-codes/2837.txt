#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const long double PI = 3.141592653589793238462643383279502884197;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	map<string, int> mp;
	mp.insert({ "AC", 0 });
	mp.insert({ "WA", 0 });
	mp.insert({ "TLE", 0 });
	mp.insert({ "RE", 0 });

	while (tc--) {
		string s; cin >> s;
		mp[s]++;
	}

	cout << "AC x " << mp["AC"] << '\n';
	cout << "WA x " << mp["WA"] << '\n';
	cout << "TLE x " << mp["TLE"] << '\n';
	cout << "RE x " << mp["RE"] << '\n';

	return 0;
}
