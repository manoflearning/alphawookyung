#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N, a, b; 
		cin >> N >> a >> b;

		string s;
		for (int i = 0; i < b; i++)
			s.push_back('a' + i);
		while (s.size() < a) s.push_back('a');

		string ans;
		while (ans.size() < N) {
			for (char x : s) ans.push_back(x);
		}

		while (ans.size() > N) ans.pop_back();

		cout << ans << endl;
	}

	return 0;
}