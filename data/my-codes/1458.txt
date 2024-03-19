#include <iostream>
#include <vector>
#include <algorithm>
#define ll long long
using namespace std;

vector<int> rate[21];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N, K;
	cin >> N >> K;

	for (int i = 1; i <= N; i++) {
		string s; cin >> s;
		rate[s.size()].push_back(i);
	}

	ll ans = 0;
	for (int i = 2; i <= 20; i++) {
		for (int l = 0; l < rate[i].size(); l++) {
			int len = upper_bound(rate[i].begin(), rate[i].end(), rate[i][l] + K) - upper_bound(rate[i].begin(), rate[i].end(), rate[i][l]);
			ans += len;
		}
	}

	cout << ans << '\n';

	return 0;
}