#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
using namespace std;

int N, H;
vector<int> down, up;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N >> H;

	for (int i = 0; i < N; i++) {
		int x; cin >> x;
		if (i % 2 == 0) down.push_back(x);
		else up.push_back(x);
	}

	sort(down.begin(), down.end());
	sort(up.begin(), up.end());

	map<int, int> mp;
	for (int i = 0; i < H; i++) {
		int dCnt = N / 2 - (lower_bound(down.begin(), down.end(), i + 0.5) - down.begin());
		int uCnt = N / 2 - (lower_bound(up.begin(), up.end(), H - (i + 0.5)) - up.begin());
		
		int cnt = dCnt + uCnt;
		
		if (mp.find(cnt) == mp.end()) mp.insert({ cnt, 1 });
		else mp.find(cnt)->second++;
	}

	cout << mp.begin()->first << ' ' << mp.begin()->second << '\n';

	return 0;
}