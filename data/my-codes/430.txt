#include <iostream>
#include <map>
#include <vector>
using namespace std;

map<int, int> mp;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	vector<int> arr(N);
	for (int& i : arr) cin >> i;

	int M; cin >> M;
	for (int i = 0; i < M; i++) {
		int x; cin >> x;
		if (mp.find(x) != mp.end()) mp.find(x)->second++;
		else mp.insert({ x, 1 });
	}

	int ans = 0;
	while (!mp.empty()) {
		ans++;

		bool canErase = false;
		for (int& i : arr) {
			if (mp.empty()) break;

			map<int, int>::iterator it = mp.lower_bound(i);

			if (it == mp.end() || i < it->first) {
				if (it == mp.begin()) continue;
				else it--;
			}
			canErase = true;
			if (it->second == 1) mp.erase(it);
			else it->second--;
		}
		if (!canErase) break;
	}

	if (!mp.empty()) cout << -1 << '\n';
	else cout << ans << '\n';

	return 0;
}