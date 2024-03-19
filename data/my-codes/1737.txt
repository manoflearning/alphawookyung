#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;

map<int, int> mp;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		mp.clear();

		int Q; cin >> Q;

		while (Q--) {
			char x; int d;
			cin >> x >> d;

			if (x == 'I') {
				if (mp.find(d) != mp.end()) mp.find(d)->second++;
				else mp.insert({ d, 1 });
			}
			else {
				if (mp.empty()) continue;
				
				map<int, int>::iterator it;
				if (d == -1) it = mp.begin();
				else {
					it = mp.end();
					it--;
				}
				
				if (it->second >= 2) it->second--;
				else mp.erase(it);
			}
		}

		if (mp.empty()) cout << "EMPTY" << '\n';
		else {
			map<int, int>::iterator it = mp.end(); it--;
			cout << it->first << ' ' << mp.begin()->first << '\n';
		}
	}

	return 0;
}