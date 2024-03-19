#include <iostream>
#include <vector>
#include <algorithm>
#define ll long long
using namespace std;

string s;
vector<int> dist;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		dist.clear();

		cin >> s;

		dist.push_back(-1);
		for (int i = 0; i < s.size(); i++) {
			if (s[i] == 'R') dist.push_back(i);
		}
		dist.push_back(s.size());

		int ans = 0;
		for (int i = 1; i < dist.size(); i++)
			ans = max(ans, dist[i] - dist[i - 1]);

		cout << ans << '\n';
	}

	return 0;
}